import os
from typing import  Union

from .models import (loaders,VAM_API_ENDPOINT,BaseObjectOrm, make_request ,AccountMixin, AssetMixin,FutureUSDMMixin,
Asset,AssetFutureUSDM,ExecutionVenue,Trade,BaseVamPydanticModel,Optional,DATE_FORMAT,AccountRiskFactors,
DoesNotExist )
import datetime
import pandas as pd
import json
from pydantic import  condecimal
from .utils import CONSTANTS, get_vam_client_logger
from .local_vault import VAULT_PATH, get_secrets_for_account_id
from cryptography.fernet import Fernet
import random

logger = get_vam_client_logger()

class BinanceBaseObject(BaseObjectOrm):
    END_POINTS = {
        "BinanceFuturesUSDMTrade": 'trade/futureusdm',
        "BinanceSpotAccount": 'account/spot',
        "BinanceFuturesAccount":  'account/futures',
        "BinanceAsset":'asset/spot',
        "BinanceAssetFutureUSDM":'asset/futureusdm'
    }
    ROOT_URL = VAM_API_ENDPOINT+"/binance"


#assets

class BinanceAsset(AssetMixin,BinanceBaseObject):
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



    def sync_funds(self,
                   target_trade_time: Union[None, datetime.datetime] = None,
                   target_holdings: Union[None, dict] = None, holdings_source: Union[str, None] = None,
                   target_weights: Union[None, dict] = None,
                   end_of_execution_time: Union[None, datetime.datetime] = None, timeout=None,
                   is_trade_snapshot=False
                   ) -> [pd.DataFrame,BinanceFuturesAccountRiskFactors]:

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





        data_for_sync = build_binance_futures_account_spanshot(api_key=self.account_api_key,
                                                               secret_key=self.account_secret_key,
                                                               orm_account=self
                                                               )
        if data_for_sync is None:
            return pd.DataFrame(),None
        risk_factors=BinanceFuturesAccountRiskFactors(**data_for_sync["account_risk_factors"])
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
        # logger.info(account_data['account_info'])
        r = make_request(s=self.build_session(), loaders=self.LOADERS, r_type="POST", url=url, payload=payload,
                         timeout=timeout)
        if r.status_code == 404:
            #asset not found Create
            response=r.json()
            raise Exception(response.json())

        if r.status_code != 200:

            raise Exception("Error Syncing funds in account")
        if target_trade_time is None:
            return pd.DataFrame() ,risk_factors

        return pd.DataFrame(json.loads(r.json())), risk_factors

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






def build_binance_futures_account_from_key(execution_venue:ExecutionVenue,api_key: str, api_secret: str,
                                           account_id:str) :
    """

    Args:
        execution_venue:
        api_key:
        api_secret:

    Returns:

    """
    import ccxt
    exchange_config={'options': {
        'defaultType': 'future',  # Ensure we are working with futures
        'defaultSubType': 'linear' #usdm
    },
        'apiKey':api_key,
        'secret': api_secret
    }

    exchange = ccxt.binance(exchange_config)
    if execution_venue.symbol in CONSTANTS.TESTNET_VENUES:
        exchange.set_sandbox_mode(True)
       
    account_balance = exchange.fetchBalance()
    cash_asset_balances = account_balance["total"]
    account_balance=account_balance['info']

    orm_account, _ = BinanceFuturesAccount.filter(account_id=account_id)

    secrets=dict(api_key=api_key, secret_key=api_secret) if VAULT_PATH is None else dict(api_key="IN_VAULT",
                                                                                         secret_key="IN_VAULT")

    patching_values = dict(**secrets,
                           multi_assets_margin=account_balance.get("multiAssetsMargin",0.0),
                           fee_burn=account_balance.get("feeBurn",False),
                           can_deposit=account_balance.get("canDeposit",False),
                           can_withdraw=account_balance.get("canWithdraw",False)
                           )

    if len(orm_account) == 0:
        existing_asset, r = BinanceAsset.filter(symbol="USDT",
                                               asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_SPOT,
                                               execution_venue__symbol=execution_venue.symbol,
                                               )
        if len(existing_asset) != 1:
            raise Exception("Binance Assets needs to be built first")
        base_values = dict( # base features
            account_id=account_id,
            execution_venue=execution_venue.id,
            account_is_active=bool(account_balance["canTrade"]),
            account_name=f"Account for {account_id}",
            cash_asset=existing_asset[0].id,


        )

        base_values.update(patching_values)
        orm_account = BinanceFuturesAccount.create(**base_values)
        orm_account,_= BinanceFuturesAccount.filter(account_id=account_id)
    assert len(orm_account)==1
    orm_account = orm_account[0]
    # patch account holdings

    orm_account.sync_funds()


def build_binance_futures_account_spanshot(api_key, secret_key,orm_account,)->Union[dict,None]:
    """

    Args:
        api_key:
        secret_key:

    Returns:

    """
    # 1 get account patch values
    POSITIONS_FILTER_COLUMNS = ["symbol", "leverage", "unrealized_pl", "strike_price", "price", "holding"]
    import ccxt
    exchange_config = {'options': {
        'defaultType': 'future',  # Ensure we are working with futures
        'defaultSubType': 'linear'  # usdm
    },
        'apiKey': api_key,
        'secret': secret_key
    }

    cash_asset_symbol=orm_account.cash_asset.symbol

    exchange = ccxt.binance(exchange_config)
    if orm_account.execution_venue.symbol in CONSTANTS.TESTNET_VENUES:
        exchange.set_sandbox_mode(True)

    account_balance = exchange.fetchBalance()
    cash_assets=account_balance["total"]
    if cash_asset_symbol not in cash_assets:
        cash_assets[cash_asset_symbol]=0.0

    account_snapshot_timestamp=account_balance["timestamp"] if account_balance["timestamp"]  is not None else datetime.datetime.utcnow().timestamp()
    account_balance = account_balance['info']
    
    
    #handle cash assets
    cash_asset_balances_df = pd.Series(cash_assets).astype(float)
    cash_asset_balances_df.index.name = "symbol"
    cash_asset_balances_df = cash_asset_balances_df.to_frame("quantity").reset_index()
    spot_prices = {}
    for a in cash_asset_balances_df.symbol:

        if a != cash_asset_symbol:
            try:
                price = exchange.fetch_ticker(a + cash_asset_symbol)['last']
            except Exception as e:
                price = 0.0
        else:
            price = 1.0
        spot_prices[a] = {"price": price,
                          "settlement_symbol": cash_asset_symbol,
                          }
    cash_asset_balances_df["price"] = cash_asset_balances_df["symbol"].apply(lambda x: spot_prices[x]["price"])
    cash_asset_balances_df["settlement_symbol"] = cash_asset_balances_df["symbol"].apply(
        lambda x: spot_prices[x]["settlement_symbol"])
    cash_asset_balances_df["asset_class"] = CONSTANTS.ASSET_TYPE_CRYPTO_SPOT

    # usdm will always settle in quote asset
    cash_assets, _ = BinanceAsset.filter(symbol__in=cash_asset_balances_df.symbol.to_list(),
                                         execution_venue__symbol=orm_account.execution_venue.symbol,
                                         )
    cash_assets_map = {a.symbol: a for a in cash_assets}
    cash_asset_balances_df["asset_id"] = cash_asset_balances_df["symbol"].map(lambda x: cash_assets_map[x].id)
    cash_asset_balances_df["settlement_asset_id"] = cash_asset_balances_df["settlement_symbol"].map(
        lambda x: cash_assets_map[x].id)
    cash_asset_balances_df = cash_asset_balances_df[cash_asset_balances_df.quantity != 0.0]
    
    account_patching_values = dict(
                           multi_assets_margin=account_balance.get("multiAssetsMargin",True),
                           fee_burn=0.0,
                           can_deposit=account_balance.get("canDeposit",False),
                           can_withdraw=account_balance.get("canWithdraw",False)
                           )


    #2 get holdings risk factors
    risk_factors = dict(account_balance=account_balance['totalWalletBalance'],
                        total_initial_margin=account_balance['totalInitialMargin'],
                        total_maintenance_margin=account_balance['totalMaintMargin'],
                        total_unrealized_profit=account_balance['totalUnrealizedProfit'],


                        total_margin_balance=account_balance['totalMarginBalance'],
                        total_cross_wallet_balance=account_balance['totalCrossWalletBalance'],
                        total_cross_unrealized_pnl=account_balance['totalCrossUnPnl'],
                        available_balance=account_balance['availableBalance'],
                        max_withdraw_amount=account_balance['maxWithdrawAmount'],
                        )
    risk_factors={k:float(v) for k,v in risk_factors.items()}



    # 3 positions &  riskfactors
    positions = exchange.fetchPositions()
    if len(positions)==0:
        positions=pd.DataFrame()
        holdings =   cash_asset_balances_df[["asset_id", "quantity", "price",
                                           "asset_class", "settlement_asset_id"]].to_dict("records")

        valuation_asset_prices=cash_asset_balances_df.set_index("asset_id")["price"].to_dict()
        missing_assets_from_futures_exposure = orm_account.get_missing_assets_in_exposure(asset_list_ids=[],
                                                                                          asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_USDM)
        missing_assets_from_cash_exposures = orm_account.get_missing_assets_in_exposure(
            asset_list_ids=[],
            asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_SPOT)
        missing_assets = missing_assets_from_futures_exposure +  missing_assets_from_cash_exposures
    else:
        positions=pd.DataFrame(positions)
    
        positions["symbol"]=positions["info"].apply(lambda x: x["symbol"])
        positions=positions.rename(columns={'unrealizedPnl':"unrealized_pl","entryPrice":"strike_price",
                                  "markPrice":"price",'liquidationPrice':"liquidation_price"})
        positions["quantity"]=positions["contracts"].astype(float)*positions["side"].map({"short":-1,"long":1})
        if positions["quantity"].isnull().sum()>0:
            raise Exception("Holding cant be Null")
        if positions['contractSize'].astype(float).max()!=1.0:
            raise Exception("quantity may not mmatch positions")
    
    
    
    
        futures_assets, _ = BinanceAssetFutureUSDM.filter(symbol__in=positions.symbol.to_list(),
                                                          execution_venue__symbol=orm_account.execution_venue.symbol,
                                                          )
        futures_assets_map = {a.symbol: a for a in futures_assets}
        positions["asset_id"] = positions["symbol"].map(lambda x: futures_assets_map[x].id)
        positions["settlement_symbol"] = positions["symbol"].map(lambda x: futures_assets_map[x].quote_asset.symbol)
        futures_settles= positions["settlement_symbol"].unique()

        #integrate cash asset balances
        # Get cash assets that are been used as colalter
        
        missing_settle_assets=[]
        for fs in futures_settles:
            if fs not in cash_assets_map.keys():
                #must look for symbol in database and add to missing assets
                tmp_asset=Asset.filter(symbol=fs,execution_venue__symbol=orm_account.execution_venue.symbol,
                                       asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_USDM
                                       )
                assert len(tmp_asset)==1, "more than one asset with search characteristics"
                missing_settle_assets.append(tmp_asset)
                

        positions["settlement_asset_id"] = positions["settlement_symbol"].map(
            lambda x: cash_assets_map[x].id)
        positions["asset_class"] = CONSTANTS.ASSET_TYPE_CRYPTO_USDM

        positions = positions[positions.quantity != 0.0]

        # fectch expected positions_from_account to guarantee there are prices, this step is critical as backend needss full price info
        valuation_asset_prices = positions.set_index("asset_id")["price"].to_dict()
        valuation_asset_prices.update(cash_asset_balances_df.set_index("asset_id")["price"].to_dict())
        positions = positions[["asset_id", "quantity", "price", "settlement_asset_id",
                               "strike_price", "unrealized_pl", "asset_class"]]
        asset_list_ids = positions["asset_id"].to_list()
        missing_assets_from_futures_exposure = orm_account.get_missing_assets_in_exposure(asset_list_ids=asset_list_ids,
                                                                    asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_USDM)
        missing_assets_from_cash_exposures = orm_account.get_missing_assets_in_exposure(asset_list_ids=[a.id for a in cash_assets_map.values()],
                                                                                          asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_SPOT)
        
        missing_assets=missing_assets_from_futures_exposure+missing_settle_assets+missing_assets_from_cash_exposures
        

        positions=positions.to_dict("records")
        holdings = positions + \
                   cash_asset_balances_df[["asset_id", "quantity", "price",
                                           "asset_class", "settlement_asset_id"]].to_dict("records")
        
    
    latest_positions=orm_account.account_target_portfolio.latest_positions
    all_expected_asset_prices=[]
    if latest_positions is not None:
        all_expected_asset_prices=[a.asset for p in orm_account.account_target_portfolio.latest_positions.positions for a in p.assets_in_position]
    missing_expected_assets_in_prices={a.id:a for a in all_expected_asset_prices if a.id not in valuation_asset_prices.keys()}
    # holdings and expected holdings needs to be accounted for
    missing_asset_prices = {}
    missing_assets=missing_assets+list(missing_expected_assets_in_prices.values())
    
    if len(missing_assets) != 0:
        futures_symbols_to_query,cash_asset_symbol_to_query={},{}
        for a in missing_assets:
            if a.asset_type in CONSTANTS.ASSET_TYPE_CRYPTO_USDM:
                futures_symbols_to_query[a.symbol]={"id": a.id, "settlement_asset_id": a.quote_asset}
            elif a.asset_type in CONSTANTS.ASSET_TYPE_CRYPTO_SPOT:
                if a.symbol==orm_account.cash_asset.symbol:
                    valuation_asset_prices[a.id]=1.0
                    continue
                cash_asset_symbol_to_query[a.symbol] = {"id": a.id, "settlement_asset_id": orm_account.cash_asset.id}
            else:
                raise NotImplementedError
                
        missing_asset_prices = exchange.fetch(
            f'https://fapi.binance.com/fapi/v1/ticker/price?symbols={",".join(futures_symbols_to_query.keys())}')
        missing_asset_prices = pd.DataFrame(missing_asset_prices)
        missing_asset_prices = missing_asset_prices[
            missing_asset_prices.symbol.isin(futures_symbols_to_query.keys())]
        missing_asset_prices["asset_id"] = missing_asset_prices["symbol"].map(
            lambda x: futures_symbols_to_query[x]["id"])
        missing_asset_prices = missing_asset_prices.set_index("asset_id")["price"].to_dict()
        # missing_assets[["quantity","unrealized_pl","asset_class"]]=[0.0,0.0,CONSTANTS.ASSET_TYPE_CRYPTO_USDM]
        # missing_assets["strike_price"]=missing_assets["price"]
        # missing_assets["settlement_asset_id"] = missing_assets["symbol"].map(lambda x: futures_symbols_to_query[x]["settlement_asset_id"])
        # positions=pd.concat([positions,missing_assets[positions.columns]],axis=0,ignore_index=True)
    valuation_asset_prices.update(missing_asset_prices)
    
    
    if len(cash_assets)==0 and len(positions)==0:
        logger.info("Account does not hold any position")
        cash_assets={orm_account.cash_asset.symbol:0.0}

    



    data_for_sync = dict(
        valuation_asset_prices=valuation_asset_prices,
        account_patching_values=account_patching_values,
        account_holdings=holdings,
        account_risk_factors=risk_factors,
        account_snapshot_timestamp=account_snapshot_timestamp,
    )
    return data_for_sync
