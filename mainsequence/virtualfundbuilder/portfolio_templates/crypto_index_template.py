from mainsequence.virtualfundbuilder.strategy_factory.signal_factory import MARKET_CAP_SIGNAL
from mainsequence.virtualfundbuilder.strategy_factory.rebalance_factory import TIME_WEIGHTED
from mainsequence.vam_client import Asset, CONSTANTS
from examples.utils import TOP_50

asset_list = [
    dict(
        symbol = f"{f}USDT",
        asset_type = CONSTANTS.ASSET_TYPE_CRYPTO_USDM
    ) for f in TOP_50
]

cash_asset, _ = Asset.filter(
    symbol="USDT",
    asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_SPOT,
    execution_venue__symbol=CONSTANTS.BINANCE_FUTURES_EV_SYMBOL
)

crypto_index_template_config = dict(
    assets_configuration=dict(
        asset_universe={CONSTANTS.BINANCE_FUTURES_EV_SYMBOL: asset_list},
        cash_asset=cash_asset[0],
        prices_configuration=dict(
            bar_frequency_id="1m",
            upsample_frequency_id="15m",
            intraday_bar_interpolation_rule="ffill",
        ),
    ),
    backtesting_weights_config = dict(
        weights_frequency = "15m",
        rebalance_strategy_name = TIME_WEIGHTED,
        rebalance_strategy_config = dict(
            calendar = '24/7',
            rebalance_start = "4pm",
            rebalance_end = '5pm',  # when to perform execution, format H:MM:SS[am,pm]
            rebalance_frequency_strategy = "daily"
        ),
        signal_weights_name = MARKET_CAP_SIGNAL,
        signal_weights_config = dict(
            source_frequency = "1d",
            num_top_assets = None,
        ),
    ),
    execution_configuration={"commission_fee": .018 / 100}  # binance maker commission
)