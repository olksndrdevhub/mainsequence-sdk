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

    def sync_funds(self,
                   target_trade_time: Union[None, datetime.datetime] = None,
                   target_holdings: Union[None, dict] = None, holdings_source: Union[str, None] = None,
                   target_weights: Union[None, dict] = None,
                   end_of_execution_time: Union[None, datetime.datetime] = None, timeout=None,
                   is_trade_snapshot=False
                   ) -> [pd.DataFrame,AlapaAccountRiskFactors]:

        """

        Parameters
        ----------
        account_data :
        target_trade_time :
        target_holdings :
        holdings_source :
        end_of_execution_time :

        Returns
        -------

        """

        data_for_sync = {}
        base_url = self.get_object_url()
        url = f"{base_url}/{self.id}/sync_funds/"
        payload = {"json": {}}
        if target_trade_time is not None:
            end_of_execution_time = end_of_execution_time.strftime(
                DATE_FORMAT) if end_of_execution_time is not None else None
            payload["json"] = {"target_trade_time": target_trade_time.strftime(DATE_FORMAT),
                               "target_holdings": target_holdings,
                               "target_weights": target_weights,
                               "holdings_source": holdings_source,
                               "end_of_execution_time": end_of_execution_time,
                               "is_trade_snapshot": is_trade_snapshot
                               }
        payload["json"].update(data_for_sync)
        r = make_request(s=self.build_session(), loaders=self.LOADERS, r_type="POST", url=url, payload=payload,
                         timeout=timeout)
        if r.status_code == 404:
            # asset not found Create
            raise Exception(r.text)

        if r.status_code != 200:
            raise Exception("Error Syncing funds in account")
        if target_trade_time is None:
            return pd.DataFrame(),risk_factors

        return pd.DataFrame(json.loads(r.json())),risk_factors


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


## Management Functions
def build_alpaca_account_from_keys(execution_venue: ExecutionVenue, api_key: str, api_secret: str,
                                   account_id: str) -> AlpacaAccount:
    """

    Args:
        execution_venue:
        api_key:
        api_secret:

    Returns:

    """

    trading_client = TradingClient(api_key, api_secret)
    account = trading_client.get_account()

    id_hex = account.id.hex

    orm_account, _ = AlpacaAccount.filter(id_hex=id_hex)

    patching_values = dict(api_key=api_key,
                           secret_key=api_secret,
                           account_number=account.account_number,
                           id_hex=account.id.hex,
                           account_blocked=account.account_blocked,
                           multiplier=float(account.multiplier),
                           options_approved_level=int(account.options_approved_level),
                           options_trading_level=int(account.options_trading_level),
                           pattern_day_trader=account.pattern_day_trader,
                           trade_suspended_by_user=account.trade_suspended_by_user,
                           trading_blocked=account.trading_blocked,
                           transfers_blocked=account.transfers_blocked,
                           shorting_enabled=account.shorting_enabled)

    if len(orm_account) == 0:
        existing_asset, r = AlpacaAsset.filter(symbol=account.currency,
                                               asset_type=CONSTANTS.ASSET_TYPE_SETTLEMENT_ASSET,
                                               execution_venue__symbol=execution_venue.symbol,
                                               )
        if len(existing_asset) != 1:
            raise Exception("Alpaca Assets needs to be built first")
        base_values = dict(  # base features
            account_id=account_id,
            execution_venue=execution_venue.id,
            account_is_active=not bool(account.account_blocked),
            account_name=f"Alpaca Account {account_id}",
            cash_asset=existing_asset[0].id, )

        base_values.update(patching_values)
        orm_account = AlpacaAccount.create(**base_values)
        orm_account, _ = AlpacaAccount.filter(account_id=account_id)

    orm_account = orm_account[0]
    # patch account holdings

    orm_account.sync_funds()



