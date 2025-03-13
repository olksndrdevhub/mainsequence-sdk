from mainsequence.virtualfundbuilder.strategy_factory.signal_factory import MOMENTUM_SIGNAL, LONG_SHORT_MOMENTUM
from mainsequence.virtualfundbuilder.strategy_factory.rebalance_factory import VOLUME_PARTICIPATION
from mainsequence.mainsequence_client import Asset, CONSTANTS
from examples.utils import TOP_50

asset_list = [
    dict(
        symbol = f"{f}USDT",
        asset_type = CONSTANTS.ASSET_TYPE_CRYPTO_USDM
    ) for f in TOP_50
]

cash_asset, _ = Asset.filter(
    symbol = "USDT",
    asset_type = CONSTANTS.ASSET_TYPE_CRYPTO_SPOT,
    execution_venue__symbol = CONSTANTS.BINANCE_FUTURES_EV_SYMBOL
)

momentum_portfolio_config = dict(
    assets_configuration = dict(
        asset_universe = {CONSTANTS.BINANCE_FUTURES_EV_SYMBOL: asset_list},
        cash_asset = cash_asset[0],
        prices_configuration = dict(
            bar_frequency_id = "1m",
            upsample_frequency_id = "15m",
            intraday_bar_interpolation_rule = "ffill",
        )
    ),
    backtesting_weights_config = dict(
        weights_frequency = "15m",
        rebalance_strategy_name = VOLUME_PARTICIPATION,
        rebalance_strategy_config = dict(
            calendar = '24/7',
            rebalance_start = "9:00",# utc dates US market hours open
            rebalance_end = '23:00',  # when to perform execution, format H:MM:SS[am,pm]
            rebalance_frequency_strategy = "daily",
            max_percent_volume_in_bar = .01,
            total_notional = 50*1e6
        ),
        signal_weights_name = MOMENTUM_SIGNAL,
        signal_weights_config = dict(
            source_frequency = "15m",  # frequency of prices
            momentum_window = "48h",
            momentum_strategy = LONG_SHORT_MOMENTUM,
            momentum_percentile = 0.15,
            signal_weights_strategy_config = dict()
        )
    ),
    execution_configuration = dict(
        commission_fee = .018 / 100
    )  # binance maker commission
)
