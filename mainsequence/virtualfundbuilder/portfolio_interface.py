import copy
from mainsequence.tdag.utils import write_yaml
from mainsequence.client import BACKEND_DETACHED
import os
from typing import Dict, Any, List, Union, Optional
import yaml
import re

from .config_handling import configuration_sanitizer
from .time_series import PortfolioStrategy
from mainsequence.client import Asset, AssetFutureUSDM, MARKETS_CONSTANTS as CONSTANTS, TargetPortfolio, Calendar

from .models import PortfolioConfiguration
from .utils import find_ts_recursively, get_vfb_logger, is_jupyter_environment
from mainsequence.client import TDAG_CONSTANTS

class PortfolioInterface():
    """
    Manages the overall strategy of investing. It initializes the tree and runs it either within the scheduler or
    directly with a full tree update.
    """

    def __init__(self, portfolio_config_template: dict, configuration_name: str=None):
        """
        Initializes the portfolio strategy with the necessary configurations.
        """
        # persist data source to pickle
        if configuration_name:
            self.check_valid_configuration_name(configuration_name)
        self.portfolio_config_template = portfolio_config_template
        self.portfolio_config = configuration_sanitizer(portfolio_config_template, auto_complete=True)
        self.configuration_name = configuration_name

        self.portfolio_tdag_update_configuration = self.portfolio_config.portfolio_tdag_update_configuration
        self.portfolio_vam_config = self.portfolio_config.portfolio_vam_configuration
        self.portfolio_build_configuration = self.portfolio_config.portfolio_build_configuration
        self.logger = get_vfb_logger()

        self._is_initialized = False

    def __str__(self):
        configuration_name = self.configuration_name or "-"
        str_configuration = yaml.dump(self.portfolio_config_template, default_flow_style=False)
        return f"Configuration Name: {configuration_name}\n{str_configuration}"

    def __repr__(self):
        return self.__str__()

    def _initialize_nodes(self) -> None:
        """
        Initializes the portfolio strategy for backtesting and for live prediction.
        Also, forces an update of the build configuration in tdag to guarantee that assets are properly rebuilt
        """
        patch = os.environ.get("PATCH_BUILD_CONFIGURATION", "False")
        os.environ[
            "PATCH_BUILD_CONFIGURATION"] = "True"  # It always needs to be true as we always want to overwrite the build
        self.portfolio_strategy_time_serie_backtest = PortfolioStrategy(
            is_live=False,
            portfolio_build_configuration=copy.deepcopy(self.portfolio_build_configuration)
        )
        self.portfolio_strategy_time_serie_live = PortfolioStrategy(
            is_live=True,
            portfolio_build_configuration=copy.deepcopy(self.portfolio_build_configuration)
        )
        os.environ["PATCH_BUILD_CONFIGURATION"] = patch
        self._is_initialized = True

    def build_target_portfolio_in_vam(self,portfolio_tags:Optional[List[str]]=None) -> TargetPortfolio:
        """
        This method creates a portfolio in VAM with configm file settings.

        Returns:
        """
        from mainsequence.client import TargetPortfolioIndexAsset
        if not self._is_initialized:
            self._initialize_nodes()

        live_ts = self.portfolio_strategy_time_serie_live
        backtest_ts = self.portfolio_strategy_time_serie_backtest

        def build_vam_portfolio(ts,build_purpose):
            from mainsequence.client import BACKEND_DETACHED

            # when is live target portfolio

            signal_weights_ts = ts.signal_weights
            
            #portfolio configuration

            #timeseries can be running in local lake so need to request  the id
            if BACKEND_DETACHED():
                standard_kwargs = dict(is_asset_only=False,
                                       local_time_serie_id=0,
                                       local_time_serie_hash_id=ts.local_hash_id,
                                       is_active=True,
                                       signal_local_time_serie_id=0,
                                       build_purpose=build_purpose,

                                       )
                #need to pickle because there is not local met
                ts.persist_to_pickle(overwrite=True)
            else:
                standard_kwargs = dict(is_asset_only=False,
                                       local_time_serie_id=ts.local_metadata.id,
                                       is_active=True,
                                       signal_local_time_serie_id=signal_weights_ts.local_metadata.id,
                                       build_purpose=build_purpose,
                                       )

            user_kwargs = self.portfolio_vam_config.model_dump()
            user_kwargs.pop("front_end_details", None)

            standard_kwargs.update(user_kwargs)
            standard_kwargs['execution_configuration']["orders_execution_configuration"]["broker_class"] = \
                standard_kwargs['execution_configuration']["orders_execution_configuration"]["broker_class"].value
            standard_kwargs["available_in_venues__symbols"] = ts.required_execution_venues_symbols
            standard_kwargs["calendar_name"]=self.portfolio_build_configuration.backtesting_weights_configuration.rebalance_strategy_configuration[
                                                        "calendar"]
            if BACKEND_DETACHED():
                standard_kwargs["available_in_venues"] = [0]
                standard_kwargs["id"] = ts.local_hash_id
                return TargetPortfolio(**standard_kwargs)

            # front end details
            standard_kwargs["target_portfolio_about"] = {
                "description": ts.get_portfolio_about_text(),
                "signal_name": ts.backtesting_weights_config.signal_weights_name,
                "signal_description": ts.signal_weights.get_explanation(),
                "rebalance_strategy_name": ts.backtesting_weights_config.rebalance_strategy_name,
            }

            standard_kwargs["backtest_table_time_index_name"] = "time_index"
            standard_kwargs["backtest_table_price_column_name"] = "portfolio"
            standard_kwargs["tags"] = portfolio_tags

            target_portfolio = TargetPortfolio.get_or_none(local_time_serie__id=ts.local_metadata.id)
            if target_portfolio is None:
                target_portfolio = TargetPortfolio.create_from_time_series(**standard_kwargs)
            else:
                # patch timeserie of portfolio to guaranteed recreation
                target_portfolio.patch(**standard_kwargs)
                self.logger.debug(f"Target portfolio {ts.local_metadata.id} already exists in VAM")

            return target_portfolio

        live_portfolio = build_vam_portfolio(live_ts, build_purpose=CONSTANTS.PORTFOLIO_BUILD_FOR_EXECUTION,)
        backtest_portfolio = build_vam_portfolio(backtest_ts, build_purpose=CONSTANTS.PORTFOLIO_BUILD_FOR_BACKTEST,)

        # create index Asset
        asset_symbol = f"{live_ts.build_prefix()}_{live_portfolio.id}_{backtest_portfolio.id}"
        
        if BACKEND_DETACHED():
            index_asset = TargetPortfolioIndexAsset(symbol=asset_symbol,
                                                    name=asset_symbol,
                                                    unique_identifier=asset_symbol,
                                                    unique_symbol=asset_symbol,
                                                    live_portfolio=live_portfolio,
                                                    backtest_portfolio=backtest_portfolio,
                                                    valuation_asset=live_ts.valuation_asset,
                                                    calendar=Calendar(
                                                    name=self.portfolio_build_configuration.backtesting_weights_configuration.rebalance_strategy_configuration[
                                                        "calendar"])
                                                    )
        else:
            index_asset = Asset.create_or_update_index_asset_from_portfolios(live_portfolio=live_portfolio.id,
                                                                             backtest_portfolio=backtest_portfolio.id,
                                                                             valuation_asset=live_ts.valuation_asset.id,
                                                                             calendar=
                                                                             self.portfolio_build_configuration.backtesting_weights_configuration.rebalance_strategy_configuration[
                                                                                 "calendar"]
                                                                             )
        self.index_asset = index_asset
        self.live_portfolio = live_portfolio
        self.backtest_portfolio = backtest_portfolio
        return live_portfolio, backtest_portfolio, index_asset

    def run(self, portfolio_tags:Optional[List[str]]=None, update_tree=True, *args, **kwargs):
        if not self._is_initialized:
            self._initialize_nodes()

        if self.portfolio_strategy_time_serie_backtest.data_source.related_resource_class_type in TDAG_CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB:
            self.portfolio_strategy_time_serie_backtest.run(debug_mode=True, update_tree=update_tree, force_update=True, **kwargs)
            self.build_target_portfolio_in_vam(portfolio_tags=portfolio_tags)
        else:
            self.portfolio_strategy_time_serie_backtest.run_local_update(*args, **kwargs)

        res = self.portfolio_strategy_time_serie_backtest.get_df_between_dates()
        if len(res) > 0:
            res = res.sort_values("time_index")
        return res

    @staticmethod
    def _connect_local_datalake(datalake_name="Default Data Lake", persist_logs_to_file=False):
        return LocalDiskSourceLake.get_or_create(datalake_name=datalake_name, persist_logs_to_file=persist_logs_to_file)

    @classmethod
    @property
    def configuration_folder_path(self):
        vfb_project_path = os.getenv("VFB_PROJECT_PATH")
        if not vfb_project_path:
            raise ValueError(
                "VFB_PROJECT_PATH environment variable is not set. "
                "Please set it before using 'configuration_path'."
            )
        return os.path.join(vfb_project_path, "configurations")

    @staticmethod
    def check_valid_configuration_name(s: str) -> bool:
        if not bool(re.match(r'^[A-Za-z0-9_]+$', s)):
            raise ValueError(f"Name {s} not valid")

    def store_configuration(self, configuration_name: Optional[str] = None):
        """
        Stores the current configuration as a YAML file under the configuration_name
        """
        if configuration_name and not self.configuration_name:
            self.configuration_name = configuration_name

        if not self.configuration_name:
            raise ValueError(
                "No configuration name was set. Provide a `configuration_name` "
                "argument or load/set one before storing."
            )

        config_file = os.path.join(
            self.configuration_folder_path,
            f"{self.configuration_name}.yaml"
        )

        write_yaml(dict_file=self.portfolio_config_template, path=config_file)
        self.logger.info(f"Configuration stored under {config_file}")
        return config_file

    @classmethod
    def load_from_configuration(cls, configuration_name,config_file:Union[str,None]=None):
        if config_file is None:
            config_file = os.path.join(cls.configuration_folder_path, f"{configuration_name}.yaml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' does not exist.")

        portfolio_config = PortfolioConfiguration.read_portfolio_configuration_from_yaml(config_file)

        portfolio = cls(portfolio_config, configuration_name)
        return portfolio

    @classmethod
    def list_configurations(cls):
        """
        Lists all YAML configuration files found in the configuration_path.
        """
        if not os.path.exists(cls.configuration_folder_path):
            return []

        files = os.listdir(cls.configuration_folder_path)
        yaml_files = [f for f in files if f.endswith(".yaml")]
        # Strip off the '.yaml' extension to return just the base names
        return [os.path.splitext(f)[0] for f in yaml_files]

    def delete_stored_configuration(self):
        """
        Removes a saved configuration file from the configuration folder
        """
        if not self.configuration_name:
            raise ValueError("No configuration name set. Cannot delete an unnamed configuration.")
        config_file = os.path.join(self.configuration_folder_path, f"{self.configuration_name}.yaml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' does not exist.")
        os.remove(config_file)
        self.logger.info(f"Deleted configuration file '{config_file}'.")


    def delete_portfolio(self):
        """
        Deletes the portfolio from vam
        :return:
        """
        #should delete
        self.live_portfolio.delete()
