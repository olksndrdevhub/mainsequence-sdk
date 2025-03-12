from typing import Literal

from .utils import make_request
from .models_vam import AccountMixin, AccountRiskFactors, CurrencyPairMixin, AssetMixin, FutureUSDMMixin, Trade
from .models_base import BaseObjectOrm, API_ENDPOINT

class BinanceBaseObject(BaseObjectOrm):
    END_POINTS = {
        "BinanceFuturesUSDMTrade": 'trade/futureusdm',
        "BinanceSpotAccount": 'account/spot',
        "BinanceFuturesAccount":  'account/futures',
        "BinanceAsset":'asset/spot',
        "BinanceAssetFutureUSDM":'asset/futureusdm',
        "BinanceCurrencyPair":'asset/currency_pair'
    }
    ROOT_URL = API_ENDPOINT + "/binance"

class BinanceAsset(AssetMixin, BinanceBaseObject):
    pass

class BinanceCurrencyPair(CurrencyPairMixin, BinanceBaseObject):
   pass

class BinanceAssetFutureUSDM(FutureUSDMMixin, BinanceBaseObject):

    @classmethod
    def batch_upsert_from_base_quote(cls, asset_config_list: list, execution_venue_symbol: str, asset_type: str,
                                     timeout=None):
        url = f"{cls.get_object_url()}/batch_upsert_from_base_quote/"
        payload = dict(json={"asset_config_list": asset_config_list, "execution_venue_symbol": execution_venue_symbol,
                             "asset_type": asset_type
                             })
        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="POST", url=url, timeout=timeout,
                         payload=payload)

        if r.status_code != 200:
            raise Exception("Error inserting assets")
        return r.json()

class BinanceFuturesAccountRiskFactors(AccountRiskFactors):
    total_initial_margin: float
    total_maintenance_margin: float
    total_margin_balance: float
    total_unrealized_profit: float
    total_cross_wallet_balance: float
    total_cross_unrealized_pnl: float
    available_balance: float
    max_withdraw_amount: float

class BaseFuturesAccount(AccountMixin, BinanceBaseObject):
    api_key :str
    secret_key :str

    multi_assets_margin: bool = False
    fee_burn: bool = False
    can_deposit: bool = False
    can_withdraw: bool = False

class BinanceFuturesTestNetAccount(BaseFuturesAccount):
    pass

class BinanceFuturesAccount(BaseFuturesAccount):
    pass

class BinanceSpotAccount(BinanceBaseObject):
    pass

class BinanceSpotTestNetAccount(BinanceBaseObject):
    pass

class BinanceEarnAccount(BinanceBaseObject):
    pass

class BinanceAssetFuturesUSDMTrade(Trade):
   pass