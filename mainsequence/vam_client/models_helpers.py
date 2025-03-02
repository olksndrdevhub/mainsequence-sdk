from typing import Union
from .models import *
from tqdm import tqdm


from .utils import CONSTANTS




def get_model_class(model_class:str):
    """
    Reverse look from model class by name
    Parameters
    ----------
    model_class

    Returns
    -------

    """
    #import inside to avouid circular reference
    from .models_alpaca import AlpacaAsset, AlpacaCurrencyPair
    from .models_binance import BinanceAsset, BinanceAssetFutureUSDM,BinanceCurrencyPair
    MODEL_CLASS_MAP={"AlpacaAsset": AlpacaAsset,
                     "AlpacaCurrencyPair": AlpacaCurrencyPair,
                     "Asset": Asset,
                     "AssetFutureUSDM": AssetFutureUSDM,
                     "BinanceAsset": BinanceAsset,
                     "BinanceAssetFutureUSDM": BinanceAssetFutureUSDM,
                     "BinanceCurrencyPair": BinanceCurrencyPair,
                     "IndexAsset": IndexAsset,
                     "TargetPortfolioIndexAsset": TargetPortfolioIndexAsset,
                     "Calendar": Calendar
                     }

    return MODEL_CLASS_MAP[model_class]
def create_from_serializer_with_class(asset_list:List[dict]):
    new_list=[]
    for a in asset_list:
        AssetClass=get_model_class(a["AssetClass"])
        a.pop("AssetClass")
        new_list.append(AssetClass(**a))
    return new_list

def module_factory(execution_venue_symbol):
    if execution_venue_symbol in CONSTANTS.BINANCE_VENUES:
        from mainsequence.vam_client import models_binance as model_module
    elif execution_venue_symbol in CONSTANTS.ALPACA_VENUES:
        from mainsequence.vam_client import models_alpaca as model_module
    elif execution_venue_symbol in CONSTANTS.NON_TRADABLE_VENUES:
        from mainsequence.vam_client import models as model_module
    else:
        raise NotImplementedError(f"Execution_venue_symbol {execution_venue_symbol} not implemented")

    return model_module
def get_right_account_class(account:Account):
    execution_venue_symbol = account.execution_venue.symbol
    model_module = module_factory(execution_venue_symbol)
    AccountClass = getattr(model_module, CONSTANTS.ACCOUNT_VENUE_FACTORY[execution_venue_symbol])
    account, _ = AccountClass.get(id=account.id)
    return account

def get_right_asset_class(execution_venue_symbol:str,asset_type:str):
    model_module = module_factory(execution_venue_symbol)
    try:
        AssetClass = getattr(model_module, CONSTANTS.ASSET_VENUE_FACTORY[execution_venue_symbol][asset_type])
    except Exception as e:
        raise Exception(f"There are no assets of type {asset_type}  in {execution_venue_symbol}")
    return AssetClass



def copy_assets_between_orms(execution_venue_symbol: Union[list,None],
                             target_orm_credentials: dict,source_orm_credentials:dict
                             , symbol_filter: Union[list, None] = None,
                             currency_filter: Union[list, None] = None, asset_type_filter: Union[list, None] = None,
                          batch_copy=50, overwrite=True,
                             ):
    from mainsequence.vam_client import CONSTANTS
    class ExecutionVenueSource(ExecutionVenue):
        authorization_headers=get_authorization_headers(**source_orm_credentials)
        ROOT_URL = source_orm_credentials["token_url"].replace("auth/rest-token-auth/", "orm/api")
        CLASS_NAME = "ExecutionVenue"
    class AssetSource(Asset):
        authorization_headers = get_authorization_headers(**source_orm_credentials)
        ROOT_URL = source_orm_credentials["token_url"].replace("auth/rest-token-auth/", "orm/api")
        CLASS_NAME = "Asset"
    class AssetFutureUSDMSource(AssetFutureUSDM):
        authorization_headers = get_authorization_headers(**source_orm_credentials)
        ROOT_URL = source_orm_credentials["token_url"].replace("auth/rest-token-auth/", "orm/api")
        CLASS_NAME = "AssetFutureUSDM"
        
    class AssetTarget(Asset):
        authorization_headers = get_authorization_headers(**target_orm_credentials)
        ROOT_URL =  target_orm_credentials["token_url"].replace("auth/rest-token-auth/", "orm/api")
        CLASS_NAME = "Asset"
   
        
    #2 Execution Venue
    if execution_venue_symbol is None:
        evs,_=ExecutionVenueSource.filter()


    for execution_venue_symbol in evs:
        #1 query Assets:
        spot, _ = AssetSource.filter(execution_venue_symbol=execution_venue_symbol,
                                    asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_SPOT)
        futures, _ = AssetFutureUSDMSource.filter(execution_venue_symbol=execution_venue_symbol,
                                     asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_USDM)
      
        assets = spot + futures
        total_assets = len(assets)
        batch_size = min(batch_copy, total_assets)
        for i in tqdm(range(0, total_assets,batch_size),desc=f"Copying Assets in {execution_venue_symbol}"):
    
            end_batch=min(i+batch_size,total_assets)
            tmp_assets=assets[i:end_batch]
    
            AssetTarget.batch_insert(asset_list=[a.serialized_config for a in tmp_assets],overwrite=overwrite)
            time.sleep(5)
        a=5