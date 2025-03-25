from typing import Dict, List, Optional

from pydantic import (BaseModel, validator, model_validator)
import os
import pandas as pd
from mainsequence.client import AssetMixin, Asset
from mainsequence.tdag.time_series import ModelList
import json
from pydantic import FieldValidationInfo, field_validator, root_validator, Field

from mainsequence.virtualfundbuilder.enums import (ExecutionVenueNames, PriceTypeNames, BrokerClassName, AssetTypes)

from mainsequence.client.models_tdag import RunConfiguration
import yaml
from mainsequence.virtualfundbuilder.utils import get_vfb_logger
from mainsequence.tdag.utils import write_yaml
from mainsequence.tdag.utils import hash_dict
import copy
from mainsequence.virtualfundbuilder.contrib.templates.asset_groups import SP500_MAP, SP500_HIGH_ESG, SP500_LOW_ESG
from functools import lru_cache

logger = get_vfb_logger()

class VFBConfigBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class PricesConfiguration(VFBConfigBaseModel):
    """
    Configuration for price data handling in a portfolio.

    Attributes:
        bar_frequency_id (str): The frequency of price bars.
        upsample_frequency_id (str): Frequency to upsample intraday data to.
        intraday_bar_interpolation_rule (str): Rule for interpolating missing intraday bars.
        is_live bool: Boolean flag indicating if the price feed is live.
    """
    bar_frequency_id: str = "1d"
    upsample_frequency_id: str = "1d"  # "15m"
    intraday_bar_interpolation_rule: str = "ffill"
    is_live: bool = False

@lru_cache(maxsize=1028)  # Cache up to 1028 different combinations
def cached_asset_filter(*args,**kwargs):

    tmp_assets = Asset.filter_with_asset_class( *args,**kwargs)
    return tmp_assets

class AssetFilter(VFBConfigBaseModel):
    """
    If the asset is a crypto coin, use 'future_usdm', if it equity, use 'cash_equity'.
    There is also a special case of asset groups that have the postfix '__group' in the asset_type, for example 'cash_equity__group.
    The allowed equity asset groups are: 'SP500', 'SP500_HIGH_ESG' (SP500 with the highest ESG scores), 'SP500_LOW_ESG' (SP500 with the lowest ESG scores) and 'US_ADR'.
    The example assets provided are only for demonstration purposes, the actual assets for the portfolio must be based on user input.
    """
    
    #Asset Properties
    asset_type: Optional[str] = None
    symbol: Optional[str] = None
    execution_venue__symbol: str
    
    #CurrencyAsset
    quote_asset__symbol: Optional[str]=None
    base_asset__categories__name:Optional[str]=None
    base_asset__categories__name_not_in:Optional[str]=None

    def parse_asset_groups(self, symbol):
        if symbol == "SP500":
            return SP500_MAP
        elif symbol == "SP500_LOW_ESG":
            return SP500_LOW_ESG
        elif symbol == "SP500_HIGH_ESG":
            return SP500_HIGH_ESG
        else:
            raise NotImplementedError(f"Unknown group '{symbol}'")

    def get_assets(self):


        return cached_asset_filter(**self.model_dump())

class AssetUniverse(VFBConfigBaseModel):
    """
    Configuration for the asset universe in a portfolio.

    Attributes:
        asset_filters (List[AssetFilter]): Filter Objects for filtering assets.

    Examples:
      asset_filters:
          - asset_type: currency_pair
            quote_asset__symbol: USDT
            execution_venue__symbol: binance
          - symbol: BTCUSDT
            asset_type: future_usdm
            execution_venue__symbol: binance
          - symbol: ETHUSDT
            asset_type: future_usdm
            execution_venue__symbol: binance
          - symbol: AAPL
            asset_type: cash_equity
            execution_venue__symbol: alpaca
          - symbol: SP500
            asset_type: cash_equity__group
          - symbol: SP500_LOW_ESG
            asset_type: cash_equity__group
    """
    asset_filters: List[AssetFilter]

    @property
    def asset_list(self) -> ModelList:
        """ Evaluates the assets in the filters and returns them as ModelList"""
        asset_list = []
        for asset_filter in self.asset_filters:
            assets = asset_filter.get_assets()
            asset_list.extend(assets)

        if len(asset_list) == 0:
            logger.warning("No asset_filters found. Returning empty ModelList.")
            return ModelList()
        return ModelList(asset_list)

    def get_assets_per_execution_venue(self):
        venue_asset_list_map = {}
        for asset in self.asset_list:
            ev_symbol = asset.execution_venue_symbol
            if ev_symbol not in venue_asset_list_map:
                venue_asset_list_map[ev_symbol] = []
            venue_asset_list_map[ev_symbol].append(asset)
        return venue_asset_list_map

    def get_filters_per_execution_venue(self):
        venue_filter_map = {}
        for filter in self.asset_filters:
            ev_symbol = filter.execution_venue__symbol
            if ev_symbol not in venue_filter_map:
                venue_filter_map[ev_symbol] = []
            venue_filter_map[ev_symbol].append(filter)
        return venue_filter_map

    def get_required_execution_venues(self):
        return list(self.get_filters_per_execution_venue().keys())

    def model_dump(self, **kwargs):
        from mainsequence.tdag.time_series.time_series import serialize_model_list
        asset_list = self.asset_list
        data = super().model_dump(**kwargs)
        data["asset_list"] = serialize_model_list(asset_list) if asset_list is not None else None
        return data

    @staticmethod
    def parse_serialized_asset_list(execution_venue_symbol: str, asset_list) -> ModelList:

        asset_universe = pd.DataFrame(asset_list)

        for (asset_type,execution_venue__symbol), asset_group in asset_universe.groupby(["asset_type","execution_venue__symbol"]):
            asset_type = asset_type[0] if isinstance(asset_type, tuple) else asset_type
            AssetClass = get_right_asset_class(asset_type=asset_type,
                                               execution_venue_symbol=execution_venue_symbol)

            tmp_assets = AssetClass.filter(
                symbol__in=asset_group["symbol"].to_list(),
                asset_type=asset_type,
                execution_venue__symbol=execution_venue__symbol,
            )

            if len(tmp_assets) != len(asset_group["symbol"].unique()):
                raise Exception(
                    f'{execution_venue_symbol} Error:These assets are not in {set(asset_group["symbol"].to_list()) - set([a.symbol for a in tmp_assets])} DB')

        return tmp_assets

    @classmethod
    def create_from_serialized_assets(cls, execution_venue_symbol: str, asset_list: list):

        """
          Parses a configuration dictionary to create a list of asset objects as a ModelList.

          Args:
              asset_configuration (Dict[str, Any]): Configuration dictionary detailing assets to parse, categorized by venue and type.

          Returns:
              ModelList: A list of assets initialized from the configuration.
          """
        asset_list = cls.parse_serialized_asset_list(execution_venue_symbol, asset_list)
        return cls(execution_venue_symbol=ExecutionVenueNames(execution_venue_symbol), asset_list=asset_list)


class AssetsConfiguration(VFBConfigBaseModel):
    """
    Configuration for assets included in a portfolio.

    Attributes:
        asset_universe (AssetUniverse):
            Asset universe categorized by execution venue.
        price_type (PriceTypeNames): Type of price used for backtesting.
        prices_configuration (PricesConfiguration): Configuration for price data handling.
    """
    asset_universe: AssetUniverse
    price_type: PriceTypeNames = PriceTypeNames.VWAP
    prices_configuration: PricesConfiguration

    def model_dump(self, **kwargs):
        "serialization is needed to speed tdag reconstruction"
        from mainsequence.tdag.time_series.time_series import serialize_model_list

        serliazed_list = serialize_model_list(self.asset_universe.asset_list)
        serialized_asset_universe = self.asset_universe.model_dump(**kwargs)
        serialized_asset_universe["asset_list"] = serliazed_list

        data = super().model_dump(**kwargs)
        data["asset_universe"] = serialized_asset_universe
        return data


class BacktestingWeightsConfig(VFBConfigBaseModel):
    """
    Configuration for backtesting weights.

    Attributes:
        rebalance_strategy_name (str): Strategy used for rebalancing.
        rebalance_strategy_configuration (Dict): Placeholder dict for the rebalance strategy configuration.
        signal_weights_name (str): Type of signal weights strategy.
        signal_weights_configuration (Dict): Placeholder dict for the signal weights configuration.
    """
    rebalance_strategy_name: str = "ImmediateSignal"
    rebalance_strategy_configuration: Dict
    signal_weights_name: str = "MarketCap"
    signal_weights_configuration: Dict

    def model_dump(self, **kwargs):
        signal_weights_configuration = self.signal_weights_configuration
        data = super().model_dump(**kwargs)
        data["signal_weights_configuration"]["signal_assets_configuration"] = signal_weights_configuration[
            "signal_assets_configuration"].model_dump(**kwargs)

        return data

    @model_validator(mode="before")
    def parse_signal_weights_configuration(cls, values):
        if isinstance(values["signal_weights_configuration"]["signal_assets_configuration"], AssetsConfiguration):
            return values

        values["signal_weights_configuration"]["signal_assets_configuration"] = AssetsConfiguration(
            asset_universe=values["signal_weights_configuration"]["signal_assets_configuration"]["asset_universe"],
            price_type=PriceTypeNames(
                values["signal_weights_configuration"]["signal_assets_configuration"]['price_type']),
            prices_configuration=PricesConfiguration(
                **values["signal_weights_configuration"]["signal_assets_configuration"]['prices_configuration'])
        )
        return values


class PortfolioExecutionConfiguration(VFBConfigBaseModel):
    """
    Configuration for portfolio execution.

    Attributes:
        commission_fee (float): Commission fee percentage.
    """
    commission_fee: float = 0.00018


class BaseRunConfiguration(RunConfiguration):
    """
    Configuration for the base dependency tree update details.

    Attributes:
        update_schedule (str): Cron-like schedule for updates.
        distributed_num_cpus (int): Number of CPUs for distributed execution.
        execution_timeout_seconds (int): Timeout for execution in seconds.
    """
    retry_on_error: int = 0
    seconds_wait_on_retry: float = 50
    required_cpus: int = 1
    required_gpus: int = 0
    execution_time_out_seconds: float = 50
    update_schedule: str = "*/1 * * * *"


class PortfolioTdagUpdateConfiguration(VFBConfigBaseModel):
    """
    Configuration for TDAG (The Data Analytics Group) updates.

    Attributes:
        base_dependency_tree_update_details (BaseDependencyTreeUpdateDetails): Details for updating dependencies.
        base_classes_to_exclude (list): List of classes to exclude from updates.
        custom_update_details_per_class (dict): Custom update details per class.
    """
    run_configuration: BaseRunConfiguration
    base_classes_to_exclude: list = list()
    custom_update_details_per_class: dict = dict()


class BrokerConfiguration(VFBConfigBaseModel):
    execution_time_out_seconds: int = 5 * 60
    max_order_life_time_seconds: int = 2


class OrderExecutionConfiguration(VFBConfigBaseModel):
    """
    Configuration for order execution.

    Attributes:
        broker_class (BrokerClassName): The class of the broker to use.
        broker_configuration (Dict): Configuration dictionary for the selected broker class.
    """
    broker_class: BrokerClassName = BrokerClassName.PRICE_CHASER_BROKER
    broker_configuration: BrokerConfiguration


class VAMExecutionConfiguration(VFBConfigBaseModel):
    """
    Configuration for Virtual Asset Management (VAM) execution.

    Attributes:
        rebalance_tolerance_percent (float): Tolerance percentage for rebalancing.
        minimum_notional_for_a_rebalance (float): Minimum notional value required for a rebalance.
        max_latency_in_cdc_seconds (int): Maximum allowed latency in seconds for CDC.
        unwind_funds_hanging_limit_seconds (int): Time limit for unwinding funds in seconds.
        minimum_positions_holding_seconds (int): Minimum time to hold positions in seconds.
        rebalance_step_every_seconds (int): Frequency of rebalance steps in seconds.
        max_data_latency_seconds (int): Maximum allowed data latency in seconds.
        orders_execution_configuration (OrderExecutionConfiguration): Configuration for order execution.
    """
    rebalance_tolerance_percent: float = 0.005
    minimum_notional_for_a_rebalance: float = 15.0
    max_latency_in_cdc_seconds: int = 300
    unwind_funds_hanging_limit_seconds: int = 3600
    minimum_positions_holding_seconds: int = 600
    rebalance_step_every_seconds: int = 300
    max_data_latency_seconds: int = 60
    orders_execution_configuration: OrderExecutionConfiguration


class PortfolioVamConfig(VFBConfigBaseModel):
    """
    Configuration for Virtual Asset Management (VAM) portfolio.

    Attributes:
        portfolio_name (str): Name of the portfolio.
        execution_configuration (VAMExecutionConfiguration): Execution configuration for VAM.
    """
    portfolio_name: str = "Portfolio Strategy Title"
    execution_configuration: VAMExecutionConfiguration  # TODO should come from VAM
    front_end_details: str = ""
    tracking_funds_expected_exposure_from_latest_holdings: bool
    builds_from_target_positions: bool

class AssetMixinOverwrite(VFBConfigBaseModel):
    """
    The Asset for evaluating the portfolio.

    Attributes:
        symbol (str): The symbol of the asset.
        execution_venue_symbol (ExecutionVenueNames): The execution venue where the asset traded. Needs to match with asset universe.
    """
    symbol: str="USD"
    execution_venue_symbol: ExecutionVenueNames=ExecutionVenueNames.ALPACA
    asset_type: AssetTypes=AssetTypes.CASH_EQUITY

class PortfolioBuildConfiguration(VFBConfigBaseModel):
    """
    Main class for configuring and building a portfolio.

    This class defines the configuration parameters needed for
    building a portfolio, including asset configurations, backtesting
    weights, and execution parameters.

    Attributes:
        assets_configuration (AssetsConfiguration): Configuration details for assets.
        portfolio_prices_frequency (str): Frequency to upsample portoflio. Optional.
        backtesting_weights_configuration (BacktestingWeightsConfig): Weights configuration used for backtesting.
        execution_configuration (PortfolioExecutionConfiguration): Execution settings for the portfolio.
        valuation_asset (AssetMixin): The Asset for evaluating the portfolio.
    """
    assets_configuration: AssetsConfiguration
    portfolio_prices_frequency: Optional[str] = "1d"
    backtesting_weights_configuration: BacktestingWeightsConfig
    execution_configuration: PortfolioExecutionConfiguration
    valuation_asset: Field(AssetMixin, portfolio_configuration_overwrite=AssetMixinOverwrite)

    def model_dump(self, **kwargs):
        serialized_asset_config = self.assets_configuration.model_dump(**kwargs)
        data = super().model_dump(**kwargs)
        data["assets_configuration"] = serialized_asset_config

        data["backtesting_weights_configuration"] = self.backtesting_weights_configuration.model_dump(**kwargs)
        data["valuation_asset"]=self.valuation_asset.to_serialized_dict()

        return data

    @root_validator(pre=True)
    def parse_assets_configuration(cls, values):

        if not isinstance(values["assets_configuration"], AssetsConfiguration) and values['assets_configuration'] is not None:
            asset_universe = values['assets_configuration']['asset_universe']
            values["assets_configuration"] = AssetsConfiguration(
                asset_universe=values['assets_configuration']['asset_universe'],
                price_type=PriceTypeNames(values['assets_configuration']['price_type']),
                prices_configuration=PricesConfiguration(
                    **values['assets_configuration']['prices_configuration'])
            )
        if not isinstance(values["valuation_asset"], AssetMixin):
            if "execution_venue" in values["valuation_asset"]:
                execution_venue_symbol = values["valuation_asset"]["execution_venue"]["symbol"]
            else:
                execution_venue_symbol = values["valuation_asset"]["execution_venue_symbol"]

            tmp_asset = Asset.get(
                symbol=values["valuation_asset"]["symbol"],
                asset_type=values["valuation_asset"]["asset_type"],
                execution_venue__symbol=execution_venue_symbol,
            )
            values["valuation_asset"] = tmp_asset

        return values


class PortfolioConfiguration(VFBConfigBaseModel):
    """
        Configuration for a complete portfolio, including build configuration,
        TDAG updates, and VAM settings.

        This class aggregates different configurations required for the
        management and operation of a portfolio.

    Attributes:
        portfolio_build_configuration (PortfolioBuildConfiguration): Configuration for building the portfolio.
        portfolio_tdag_update_configuration (PortfolioTdagUpdateConfiguration): TDAG update configuration.
        portfolio_vam_configuration (PortfolioVamConfig): VAM execution configuration.
    """
    portfolio_build_configuration: PortfolioBuildConfiguration
    portfolio_tdag_update_configuration: PortfolioTdagUpdateConfiguration
    portfolio_vam_configuration: PortfolioVamConfig

    @staticmethod
    def read_portfolio_configuration_from_yaml(yaml_path: str):
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def parse_portfolio_configuration_from_yaml(yaml_path: str, auto_complete=False):
        from mainsequence.virtualfundbuilder.config_handling import configuration_sanitizer
        configuration = PortfolioConfiguration.read_portfolio_configuration_from_yaml(yaml_path)
        return configuration_sanitizer(configuration, auto_complete=auto_complete)

    @staticmethod
    def parse_portfolio_configurations(
        portfolio_build_configuration: dict,
        portfolio_tdag_update_configuration: dict,
        portfolio_vam_configuration: dict,
    ):
        # Parse the individual components
        backtesting_weights_configuration = BacktestingWeightsConfig(
            rebalance_strategy_name=portfolio_build_configuration['backtesting_weights_configuration']['rebalance_strategy_name'],
            rebalance_strategy_configuration=portfolio_build_configuration['backtesting_weights_configuration'][
                'rebalance_strategy_configuration'],
            signal_weights_name=portfolio_build_configuration['backtesting_weights_configuration'][
                'signal_weights_name'],
            signal_weights_configuration=portfolio_build_configuration['backtesting_weights_configuration'][
                'signal_weights_configuration']
        )

        execution_configuration = PortfolioExecutionConfiguration(
            commission_fee=portfolio_build_configuration['execution_configuration']['commission_fee']
        )

        portfolio_build_config = PortfolioBuildConfiguration(
            assets_configuration=portfolio_build_configuration['assets_configuration'],
            backtesting_weights_configuration=backtesting_weights_configuration,
            execution_configuration=execution_configuration,
            portfolio_prices_frequency=portfolio_build_configuration['portfolio_prices_frequency'],
            valuation_asset=portfolio_build_configuration["valuation_asset"]
        )

        run_configuration = BaseRunConfiguration(
            **portfolio_tdag_update_configuration['base_dependency_tree_update_details']
        )

        portfolio_tdag_update_config = PortfolioTdagUpdateConfiguration(
            run_configuration=run_configuration,
            base_classes_to_exclude=portfolio_tdag_update_configuration['base_classes_to_exclude'],
            custom_update_details_per_class=portfolio_tdag_update_configuration['custom_update_details_per_class']
        )

        vam_execution_config = VAMExecutionConfiguration(
            rebalance_tolerance_percent=portfolio_vam_configuration['execution_configuration'][
                'rebalance_tolerance_percent'],
            minimum_notional_for_a_rebalance=portfolio_vam_configuration['execution_configuration'][
                'minimum_notional_for_a_rebalance'],
            max_latency_in_cdc_seconds=portfolio_vam_configuration['execution_configuration'][
                'max_latency_in_cdc_seconds'],
            unwind_funds_hanging_limit_seconds=portfolio_vam_configuration['execution_configuration'][
                'unwind_funds_hanging_limit_seconds'],
            minimum_positions_holding_seconds=portfolio_vam_configuration['execution_configuration'][
                'minimum_positions_holding_seconds'],
            rebalance_step_every_seconds=portfolio_vam_configuration['execution_configuration'][
                'rebalance_step_every_seconds'],
            max_data_latency_seconds=portfolio_vam_configuration['execution_configuration']['max_data_latency_seconds'],
            orders_execution_configuration=OrderExecutionConfiguration(
                broker_class=BrokerClassName(
                    portfolio_vam_configuration['execution_configuration']['orders_execution_configuration'][
                        'broker_class']),
                broker_configuration=
                portfolio_vam_configuration['execution_configuration']['orders_execution_configuration'][
                    'broker_configuration']
            )
        )
        portfolio_vam_configuration["execution_configuration"] = vam_execution_config
        portfolio_vam_configuration["front_end_details"] = portfolio_vam_configuration['front_end_details']["about_text"]
        portfolio_vam_configuration = PortfolioVamConfig(**portfolio_vam_configuration)

        # Combine everything into the final PortfolioConfiguration
        portfolio_config = PortfolioConfiguration(
            portfolio_build_configuration=portfolio_build_config,
            portfolio_tdag_update_configuration=portfolio_tdag_update_config,
            portfolio_vam_configuration=portfolio_vam_configuration
        )

        return portfolio_config

    def build_yaml_configuration_file(self):
        signal_type = self.portfolio_build_configuration.backtesting_weights_configuration.signal_weights_name
        vfb_folder = os.path.join(os.path.expanduser("~"), "VirtualFundBuilder", "configurations")
        vfb_folder = os.path.join(vfb_folder, signal_type)
        if not os.path.exists(vfb_folder):
            os.makedirs(vfb_folder)

        config_hash = hash_dict(self.model_dump_json())
        config_file_name = f"{vfb_folder}/{config_hash}.yaml"

        write_yaml(dict_file=json.loads(self.model_dump_json()), path=config_file_name)
        return config_file_name