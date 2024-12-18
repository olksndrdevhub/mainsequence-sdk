import os
from typing import Union, Literal
from .models import (loaders, VAM_API_ENDPOINT, BaseObjectOrm, make_request, AccountMixin, AssetMixin,
                     FutureUSDMMixin,BaseVamPydanticModel,DATE_FORMAT,
                     Asset, AssetFutureUSDM, ExecutionVenue,AccountRiskFactors,
                     DoesNotExist)
import datetime
import pandas as pd

from .utils import CONSTANTS, get_vam_client_logger
from cryptography.fernet import Fernet
from pydantic import  condecimal
from alpaca.trading.client import TradingClient

from alpaca.trading.enums import AssetClass as AlpacaAssetClass, PositionSide
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest
from .local_vault import VAULT_PATH, get_secrets_for_account_id

logger = get_vam_client_logger()

class AlpacaBaseObject(BaseObjectOrm):
    END_POINTS = {
        "AlpacaAssetTrade": 'trade/spot',
        "AlpacaAccount": 'account',
        "AlpacaAsset": 'asset/spot',
    }
    ROOT_URL = VAM_API_ENDPOINT + "/alpaca"


class AlpacaAsset(AssetMixin, AlpacaBaseObject):
    ticker: str
    asset_class: str
    exchange: str
    status: Literal["active", "inactive"]
    marginable: bool
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool
    settlement_asset_symbol: str

    def get_spot_reference_asset_symbol(self):

        return self.symbol

    def get_settlement_asset_symbol(self):
        return self.settlement_asset_symbol

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

        data_for_sync = make_alpaca_account_snapshot(api_key=self.account_api_key,
                                                     secret_key=self.account_secret_key,
                                                     orm_account=self)

        risk_factors = AlapaAccountRiskFactors(**data_for_sync["account_risk_factors"])

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




def make_alpaca_account_snapshot(api_key, secret_key, orm_account):
    """
    Makes a snapshot of account, position and riskfactors
    Parameters
    ----------
    api_key
    secret_key

    Returns
    -------

    """
    trading_client = TradingClient(api_key, secret_key)
    stock_data_client = StockHistoricalDataClient(api_key, secret_key)
    crypto_data_clientt=CryptoHistoricalDataClient(api_key, secret_key)
    account = trading_client.get_account()
    account_snapshot_timestamp = datetime.datetime.utcnow().timestamp()
    # 1 get account patch values
    account_patching_values = dict(
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

    # 2 get holdings risk_factors
    risk_factors = dict(account_balance=account.equity,
                        last_equity=account.last_equity,
                        buying_power=account.buying_power,
                        cash=account.cash,
                        total_initial_margin=account.initial_margin,
                        total_maintenance_margin=account.maintenance_margin,
                        last_maintenance_margin=account.last_maintenance_margin,
                        long_market_value=account.long_market_value,
                        non_marginable_buying_power=account.non_marginable_buying_power,
                        options_buying_power=account.options_buying_power,
                        portfolio_value=account.portfolio_value,
                        regt_buying_power=account.regt_buying_power,
                        sma=account.sma,
                        )
    # 3 positions & riskfactors
    positions = trading_client.get_all_positions()
    positions = pd.DataFrame([p.__dict__ for p in positions]).rename(columns={"asset_id": "alpaca_id"})
    positions["asset_type"] = positions["asset_class"].map({AlpacaAssetClass.CRYPTO: CONSTANTS.ASSET_TYPE_CRYPTO_SPOT,
                                                            AlpacaAssetClass.US_EQUITY: CONSTANTS.ASSET_TYPE_CASH_EQUITY,
                                                            })

    positions_assets, _ = AlpacaAsset.filter(symbol__in=positions["symbol"].to_list(),
                                             execution_venue__symbol=orm_account.execution_venue.symbol,
                                             asset_type__in=[CONSTANTS.ASSET_TYPE_CRYPTO_SPOT,CONSTANTS.ASSET_TYPE_CASH_EQUITY]
                                             )

    cash_asset, _ = AlpacaAsset.filter(symbol=account.currency,asset_type=CONSTANTS.ASSET_TYPE_SETTLEMENT_ASSET,
                                       execution_venue__symbol=orm_account.execution_venue.symbol,
                                       )

    positions_assets = positions_assets + cash_asset
    cash_asset_df = pd.DataFrame(index=[-1],
                                 data={"alpaca_id": account.currency, "avg_entry_price": 1, "qty": account.cash,
                                       "side": PositionSide.LONG if float(account.cash) > 0 else PositionSide.SHORT,
                                       "market_value": account.cash, "cost_basis": 1.0,
                                        "symbol":account.currency,
                                       "asset_type":CONSTANTS.ASSET_TYPE_SETTLEMENT_ASSET,
                                       "unrealized_pl": 0.0, "unrealized_plpc": 0.0, "current_price": 1.0})
    positions = pd.concat([positions, cash_asset_df], axis=0)
    positions["symbol_type"]=positions["symbol"]+"_"+positions["asset_type"]





    positions_assets_map = {f"{a.symbol}_{a.asset_type}": {"id": int(a.id), "asset_type": a.asset_type,
                                       "settlement_asset_symbol":a.settlement_asset_symbol,
                                       } for a in positions_assets}


    if len(positions_assets) != len(positions):
        missing_position_assets = positions[~positions.alpaca_id.isin(positions_assets_map.keys())]
        raise Exception("Alpaca Assets needs to be built first")


    positions = positions[["symbol_type", "qty", "side",
                           "unrealized_pl", "current_price"]].rename(
        columns={"qty": "quantity", "current_price": "price"})
    positions["asset_id"] = positions["symbol_type"].map(lambda x: positions_assets_map[x]["id"])
    positions["asset_class"] = positions["symbol_type"].map(lambda x: positions_assets_map[x]["asset_type"])
    positions["settlement_asset_symbol"] = positions["symbol_type"].map(lambda x: positions_assets_map[x]["settlement_asset_symbol"])
    if all(positions["settlement_asset_symbol"]==orm_account.cash_asset.symbol)==False:
        raise Exception("This asset settlement asset id differs")
    positions["settlement_asset_id"]=cash_asset[0].id

    positions = positions[positions.asset_id.isnull() == False]
    positions["side"] = positions["side"].apply(lambda x: 1.0 if x == PositionSide.LONG else -1.0)
    positions["quantity"] = positions["quantity"].astype(float) * positions["side"].astype(float)

    positions = positions.drop(columns=["side",])
    valuation_asset_prices = positions.set_index("asset_id")["price"].to_dict()
    # build by asset_type
    missing_asset_prices = {}
    for asset_type, asset_type_df in positions.groupby("asset_class"):
        asset_list_ids = asset_type_df["asset_id"].to_list()
        missing_assets = orm_account.get_missing_assets_in_exposure(asset_list_ids=asset_list_ids,
                                                                    asset_type=asset_type)
        symbols_to_query = {a["symbol"]: {"id": a["id"], "settlement_asset_id": a["quote_asset"]["id"]} for
                            a in
                            missing_assets}
        if len(missing_assets) > 0:
            if asset_type==CONSTANTS.ASSET_TYPE_CASH_EQUITY:
                latest_quotes=stock_data_client.get_stock_latest_quote(
                    request_params=StockLatestQuoteRequest(symbol_or_symbols=symbols_to_query.keys()))
            elif  asset_type==CONSTANTS.ASSET_TYPE_CRYPTO_SPOT:
                latest_quotes = crypto_data_clientt.get_stock_latest_quote(
                    request_params=CryptoLatestQuoteRequest(symbol_or_symbols=symbols_to_query.keys()))
            else:
                raise NotImplementedError
            tmp_missing_asset_prices=pd.concat([pd.Series(p.__dict__) for k,p in latest_quotes.items()],axis=1).T
            tmp_missing_asset_prices["asset_id"] = tmp_missing_asset_prices["symbol"].map(
                lambda x: futures_symbols_to_query[x]["id"])
            tmp_missing_asset_prices["price"]=(tmp_missing_asset_prices["bid_price"]+tmp_missing_asset_prices["ask_price"])/2
            tmp_missing_asset_prices = tmp_missing_asset_prices.set_index("asset_id")["price"].to_dict()
            missing_asset_prices.update(tmp_missing_asset_prices)
    #

    holdings = positions[["asset_id", "quantity", "price",
                                       "asset_class", "settlement_asset_id"]].to_dict("records")
    valuation_asset_prices.update(missing_asset_prices)

    data_for_sync = dict(
        valuation_asset_prices=valuation_asset_prices,
        account_patching_values=account_patching_values,
        account_holdings=holdings,
        account_risk_factors=risk_factors,
        account_snapshot_timestamp=account_snapshot_timestamp,

    )
    return data_for_sync
