from enum import Enum
from mainsequence.mainsequence_client import VAM_CONSTANTS as CONSTANTS

class RebalanceFrequencyStrategyName(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class BrokerClassName(Enum):
    # TODO get from VAM
    PRICE_CHASER_BROKER = "PriceChaserBroker"
    MARKET_ONLY = "MarketOnly"

ExecutionVenueNames = Enum("ExecutionVenueNames", {key.upper(): key for key in CONSTANTS.EXECUTION_VENUES_NAMES.keys()})
AssetTypes = Enum("AssetTypes", {key.upper(): key for key in CONSTANTS.ASSET_TYPES})

class PriceTypeNames(Enum):
    VWAP = "vwap"
    OPEN = "open"
    CLOSE = "close"

class RunStrategy(Enum):
    BACKTEST = "backtest"
    LIVE = "live"
    ALL = "all"


class StrategyType(Enum):
    SIGNAL_WEIGHTS_STRATEGY = "signal_weights_strategy"
    REBALANCE_STRATEGY = "rebalance_strategy"
