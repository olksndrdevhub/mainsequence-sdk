import ast

from mainsequence.tdag.time_series import TimeSerie
from datetime import datetime, timedelta
import numpy as np
import pytz

from mainsequence.virtualfundbuilder.enums import StrategyType
from mainsequence.virtualfundbuilder.strategy_factory.base_factory import BaseStrategy, BaseFactory, insert_in_registry
from mainsequence.virtualfundbuilder.models import AssetsConfiguration

import pandas as pd
from mainsequence.vam_client import (Asset, ExecutionPositions)
from mainsequence.virtualfundbuilder.utils import get_vfb_logger,_send_strategy_to_registry

logger = get_vfb_logger()

def send_weights_as_position_to_vam(method):
    MIN_DATE_FOR_HISTORICAL_WEIGHTS = datetime(2024, 8, 30).replace(tzinfo=pytz.UTC)

    def wrapper(self, update_statistics):
        signal_weights_df = method(self, update_statistics)
        if len(signal_weights_df) == 0 or self.send_weights_for_execution_to_vam == False:
            return signal_weights_df

        try:

            """
            if weights are send for execution to VAM all the portfolios that have this hash id should get an updated 
            weights
            
            When portfolios are not "LIVE" and are been used for backtesting then only Positions should be created with no execution
            
            """
            target_weights = signal_weights_df[
                signal_weights_df.index.get_level_values("time_index") >= MIN_DATE_FOR_HISTORICAL_WEIGHTS]

            # do not send repeated weights
            target_weights = target_weights[
                target_weights.index.get_level_values(
                    "time_index") == target_weights.index.get_level_values(
                    "time_index").max()]
            positions_time = target_weights.index[0][0]

            if latest_value is not None:
                if positions_time < latest_value:
                    return signal_weights_df

            target_weights = target_weights.reset_index().drop(columns=["time_index"]).rename(
                columns={"signal_weight": "weight_notional_exposure",
                         "asset_symbol": "asset_id"
                         }).dropna()

            for ev in target_weights.execution_venue_symbol.unique():
                ev_index = target_weights[target_weights['execution_venue_symbol'] == ev].index
                target_weights.loc[ev_index, "asset_id"] = target_weights["asset_id"].map(self.symbol_to_id_map[ev])

            target_weights = target_weights.drop(columns="execution_venue_symbol")
            # many can be following the same signals
            r = ExecutionPositions.add_from_time_serie(
                time_serie_signal_hash_id=self.hash_id,
                positions_time=positions_time,
                positions_list=target_weights.to_dict('records')
            )

        except Exception as e:
            self.logger.exception(f"Couldn't send weights to VAM - error {e}")
            raise e
        return signal_weights_df

    return wrapper

class PrivateWeightsBaseArguments:
    def __init__(self):
        self.send_weights_for_execution_to_vam =  self.assets_configuration.prices_configuration.is_live

class WeightsBase(PrivateWeightsBaseArguments, BaseStrategy):
    TYPE = StrategyType.SIGNAL_WEIGHTS_STRATEGY

    def __init__(self,
                 signal_assets_configuration: AssetsConfiguration,
                 *args, **kwargs):
        """
        Base Class for all signal weights

        Attributes:
            assets_configuration (AssetsConfiguration): Configuration details for signal assets.
        """
        
        self.assets_configuration = signal_assets_configuration
        self.asset_universe = signal_assets_configuration.asset_universe

        super().__init__()



    def get_explanation(self):
        info = f"""
        <p>{self.__class__.__name__}: Signal weights class.</p>
        """
        return info

    def maximum_forward_fill(self) -> timedelta:
        raise NotImplementedError

    def interpolate_index(self, new_index: pd.DatetimeIndex):
        """
        Get interpolated weights for a time index. Weights are only valid for a certain time, therefore forward fill is limited.
        """
        # get values between new index
        try:
            weights = self.get_df_between_dates(start_date=new_index.min(), end_date=new_index.max())
        except Exception as e:
            raise e
        
        # if we need more data before to interpolate first value of new_index
        if len(weights) == 0 or (weights.index.get_level_values("time_index").min() > new_index.min()):
 
            last_observation = self.get_last_observation()
            if last_observation is None:
                return pd.DataFrame()
            last_date = last_observation.index.get_level_values("time_index")[0]

            if last_date < new_index.min():
                self.logger.warning(f"No weights data at start of the portfolio at { new_index.min()}"
                                    f" will use last available weights {last_date}")
                weights = self.get_df_between_dates(start_date=last_date, end_date=new_index.max())


        if len(weights) == 0 :
            self.logger.warning(f"No weights data in index interpolation")
            return pd.DataFrame()

        weights_pivot = weights.reset_index().pivot(index="time_index", columns=[ "unique_identifier"],
                                      values="signal_weight").fillna(0)
        weights_pivot["last_weights"] = weights_pivot.index.get_level_values(level="time_index")

        # combine existing index with new index
        combined_index = weights_pivot.index.union(new_index)
        combined_index.name = "time_index"
        weights_reindex = weights_pivot.reindex(combined_index)

        # check which dates are outside of valid forward filling range
        weights_reindex["last_weights"] = weights_reindex["last_weights"].ffill()
        weights_reindex["diff_to_last_weights"] = weights_reindex.index.get_level_values(level="time_index") - \
                                                  weights_reindex["last_weights"]

        invalid_forward_fills = weights_reindex[
                                    "diff_to_last_weights"] >= self.maximum_forward_fill()  # source_frequency is the duration a weight is valid
        weights_reindex.drop(columns=["last_weights", "diff_to_last_weights"], inplace=True)

        # forward fill and set dates that are outside of valid range to nan
        weights_reindex = weights_reindex.ffill()
        weights_reindex[invalid_forward_fills] = np.nan

        if weights_reindex.isna().values.any():
            self.logger.info(f"Could not fully interpolate for signal weights")

        weights_reindex = weights_reindex.loc[new_index]
        weights_reindex.index.name = "time_index"
        return weights_reindex




def _get_class_source_code(cls):
    import ast
    import inspect
    import sys

    try:
        # Get the source code of the module where the class is defined
        module = sys.modules[cls.__module__]
        source = inspect.getsource(module)
    except Exception as e:
        logger.warning(f"Could not get source code for module {cls.__module__}: {e}")
        return None

    # Parse the module's source code
    try:
        module_ast = ast.parse(source)
        class_source_code = None

        # Iterate through the module's body to find the class definition
        for node in module_ast.body:
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                # Get the lines corresponding to the class definition
                lines = source.splitlines()
                # Get the lines for the class definition
                class_source_lines = lines[node.lineno - 1: node.end_lineno]
                class_source_code = '\n'.join(class_source_lines)
                break

        if not class_source_code:
            logger.warning(f"Class definition for {cls.__name__} not found in module {cls.__module__}")
            return None

        return class_source_code

    except Exception as e:
        logger.warning(f"Could not parse source code for module {cls.__module__}: {e}")
        return None


SIGNAL_CLASS_REGISTRY = SIGNAL_CLASS_REGISTRY if 'SIGNAL_CLASS_REGISTRY' in globals() else {}
def register_signal_class(name=None, register_in_agent=True):
    """
    Decorator to register a model class in the factory.
    If `name` is not provided, the class's name is used as the key.
    """
    def decorator(cls):
        return insert_in_registry(SIGNAL_CLASS_REGISTRY, cls, register_in_agent, name)
    return decorator


class SignalWeightsFactory(BaseFactory):
    @staticmethod
    def get_signal_weights_strategy(signal_weights_name) -> TimeSerie:
        """
        Creates an instance of the appropriate SignalWeights class based on the provided name.
        """
        if signal_weights_name not in SIGNAL_CLASS_REGISTRY:
            SignalWeightsFactory.get_signal_weights_strategies()

        return SIGNAL_CLASS_REGISTRY[signal_weights_name]

    @staticmethod
    def get_signal_weights_strategies():
        """
        Scans the given directory for Python files, imports the classes,
        and returns all classes that are subclasses of WeightsBase.
        """
        SignalWeightsFactory.import_module("signals")
        return SIGNAL_CLASS_REGISTRY