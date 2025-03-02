import os
from typing import Union, Literal
from .models import (loaders, VAM_API_ENDPOINT, BaseObjectOrm, make_request, AccountMixin, AssetMixin,
                     FutureUSDMMixin, BaseVamPydanticModel, DATE_FORMAT,
                     Asset, AssetFutureUSDM, ExecutionVenue, AccountRiskFactors,
                     DoesNotExist, CurrencyPairMixin)
import datetime
import pandas as pd
from mainsequence.logconf import logger
from .utils import CONSTANTS
from cryptography.fernet import Fernet
from pydantic import  condecimal



from .local_vault import VAULT_PATH, get_secrets_for_account_id



class AlpacaBaseObject(BaseObjectOrm):
    END_POINTS = {
        "AlpacaAssetTrade": 'trade/spot',
        "AlpacaAccount": 'account',
        "AlpacaAsset": 'asset/spot',
        "AlpacaCurrencyPair": 'asset/currency_pair',
    }
    ROOT_URL = VAM_API_ENDPOINT + "/alpaca"


class AlpacaAssetMixin(AssetMixin, AlpacaBaseObject):
    ticker: str
    asset_class: str
    exchange: str
    status: Literal["active", "inactive"]
    marginable: bool
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool

    def get_spot_reference_asset_symbol(self):
        return self.symbol

    @staticmethod
    def get_properties_from_unique_symbol(unique_symbol: str):
        if unique_symbol.endswith(CONSTANTS.ALPACA_CRYPTO_POSTFIX):
            return {"symbol": unique_symbol.replace(CONSTANTS.ALPACA_CRYPTO_POSTFIX, ""), "asset_type": CONSTANTS.ASSET_TYPE_CRYPTO_SPOT}

        return {"symbol": unique_symbol, "asset_type": CONSTANTS.ASSET_TYPE_CASH_EQUITY}

class AlpacaAsset(AlpacaAssetMixin):
    pass

class AlpacaCurrencyPair(AlpacaAssetMixin, CurrencyPairMixin):
    pass

class AlapaAccountRiskFactors(AccountRiskFactors):
    total_initial_margin: float
    total_maintenance_margin: float
    last_equity: float
    buying_power: float
    cash: float
    last_maintenance_margin: float
    long_market_value: float
    non_marginable_buying_power: float
    options_buying_power: float
    portfolio_value:float
    regt_buying_power: float
    sma: float

class AlpacaAccount(AccountMixin,AlpacaBaseObject):
    api_key: str
    secret_key: str

    account_number: str
    id_hex: str
    account_blocked: bool
    multiplier: float
    options_approved_level: int
    options_trading_level: int
    pattern_day_trader: bool
    trade_suspended_by_user: bool
    trading_blocked: bool
    transfers_blocked: bool
    shorting_enabled: bool



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
        secrets = self.get_secrets_from_local_vault()
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



# trades
class AlpacaAssetTrade(AlpacaBaseObject):

    @classmethod
    def create_or_update(cls, timeout=None, *args, **kwargs, ):
        url = f"{cls.get_object_url()}/create_or_update"
        data = cls.serialize_for_json(kwargs)
        payload = {"json": data}
        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="POST", url=f"{url}/",
                         timeout=timeout,
                         payload=payload)
        if r.status_code not in [201, 200]:
            raise Exception(r.text)
        return cls(**r.json())



