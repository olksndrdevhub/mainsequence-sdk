from .models_vam import *
from .models_binance import *
from .models_alpaca import *
from .models_base import VAM_CONSTANTS

def get_model_class(model_class: str):
    """
    Reverse look from model class by name
    """
    MODEL_CLASS_MAP = {
        "AlpacaAsset": AlpacaAsset,
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

def create_from_serializer_with_class(asset_list: List[dict]):
    new_list = []
    for a in asset_list:
        AssetClass = get_model_class(a["AssetClass"])
        a.pop("AssetClass")
        new_list.append(AssetClass(**a))
    return new_list

def module_factory(execution_venue_symbol):
    if execution_venue_symbol in VAM_CONSTANTS.BINANCE_VENUES:
        from mainsequence.mainsequence_client import models_binance as model_module
    elif execution_venue_symbol in VAM_CONSTANTS.ALPACA_VENUES:
        from mainsequence.mainsequence_client import models_alpaca as model_module
    elif execution_venue_symbol in VAM_CONSTANTS.NON_TRADABLE_VENUES:
        from mainsequence.mainsequence_client import models_vam as model_module
    else:
        raise NotImplementedError(f"Execution_venue_symbol {execution_venue_symbol} not implemented")
    return model_module

def get_right_account_class(account: Account):
    execution_venue_symbol = account.execution_venue.symbol
    model_module = module_factory(execution_venue_symbol)
    AccountClass = getattr(model_module, VAM_CONSTANTS.ACCOUNT_VENUE_FACTORY[execution_venue_symbol])
    account, _ = AccountClass.get(id=account.id)
    return account

def get_right_asset_class(execution_venue_symbol:str, asset_type:str):
    model_module = module_factory(execution_venue_symbol)
    try:
        AssetClass = getattr(model_module, VAM_CONSTANTS.ASSET_VENUE_FACTORY[execution_venue_symbol][asset_type])
    except Exception as e:
        raise Exception(f"There are no assets of type {asset_type} in {execution_venue_symbol}")
    return AssetClass