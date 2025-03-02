import os
from typing import  Union

from .models import (loaders,VAM_API_ENDPOINT,BaseObjectOrm, make_request ,AccountMixin, AssetMixin,FutureUSDMMixin,
Asset,AssetFutureUSDM,ExecutionVenue,Trade,BaseVamPydanticModel,Optional,DATE_FORMAT,AccountRiskFactors,
CurrencyPairMixin,
DoesNotExist )
import datetime
import pandas as pd
import json
from pydantic import  condecimal
from .utils import CONSTANTS
from .local_vault import VAULT_PATH, get_secrets_for_account_id
from cryptography.fernet import Fernet
import random
from mainsequence.logconf import logger

class BinanceBaseObject(BaseObjectOrm):
    END_POINTS = {
        "BinanceFuturesUSDMTrade": 'trade/futureusdm',
        "BinanceSpotAccount": 'account/spot',
        "BinanceFuturesAccount":  'account/futures',
        "BinanceAsset":'asset/spot',
        "BinanceAssetFutureUSDM":'asset/futureusdm',
        "BinanceCurrencyPair":'asset/currency_pair'
    }
    ROOT_URL = VAM_API_ENDPOINT+"/binance"


#assets

class BinanceAsset(AssetMixin,BinanceBaseObject):
    pass

class BinanceCurrencyPair(CurrencyPairMixin,BinanceBaseObject):
   pass


class BinanceAssetFutureUSDM(FutureUSDMMixin,BinanceBaseObject):

    @classmethod
    def batch_upsert_from_base_quote(cls, asset_config_list: list, execution_venue_symbol: str, asset_type: str,
                                     timeout=None):
        """

        Parameters
        ----------
        asset_config_list
        execution_venue_symbol
        asset_type

        Returns
        -------

        """

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

#accounts
class BaseFuturesAccount(AccountMixin,BinanceBaseObject):

    api_key :str
    secret_key :str

    multi_assets_margin: bool = False
    fee_burn: bool = False
    can_deposit: bool = False
    can_withdraw: bool = False



    def get_secrets_from_local_vault(self):
        if hasattr(self,"_secrets"):
            return self._secrets
        if VAULT_PATH is not None:
            secrets = get_secrets_for_account_id(self.account_id)
            self._secrets=secrets["secrets"]
        else:
            return None
        return self._secrets
    @property
    def fernet_key(self):
        fernet_key = os.environ["ACCOUNT_SETTINGS_ENCRYPTION_KEY"]
        fernet_key = Fernet(fernet_key)
        return fernet_key

    @property
    def account_api_key(self):
        secrets=self.get_secrets_from_local_vault()
        if secrets is not None:
            return secrets['api_key']

        return self.fernet_key.decrypt(self.api_key).decode(
                "utf-8")

    @property
    def account_secret_key(self):
        secrets = self.get_secrets_from_local_vault()
        if secrets is not None:
            return secrets['secret_key']
        return self.fernet_key.decrypt(self.secret_key).decode(
            "utf-8")




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

#trades
class BinanceAssetFuturesUSDMTrade(Trade):

   pass










