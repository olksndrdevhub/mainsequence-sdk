from .models_vam import *
from .base import VAM_CONSTANTS
from .models_tdag import LocalTimeSerie

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

def get_right_account_class(account: Account):
    from mainsequence.client import models_vam as model_module
    execution_venue_symbol = account.execution_venue.symbol
    AccountClass = getattr(model_module, VAM_CONSTANTS.ACCOUNT_VENUE_FACTORY[execution_venue_symbol])
    account, _ = AccountClass.get(id=account.id)
    return account

def get_right_asset_class(execution_venue_symbol:str, asset_type:str):
    from mainsequence.client import models_vam as model_module
    try:
        AssetClass = getattr(model_module, VAM_CONSTANTS.ASSET_VENUE_FACTORY[execution_venue_symbol][asset_type])
    except Exception as e:
        raise Exception(f"There are no assets of type {asset_type} in {execution_venue_symbol}")
    return AssetClass


class TDAGAPIDataSource(BaseObjectOrm, BasePydanticModel):
    id: Optional[int]
    unique_identifier: str = Field(..., description="Unique identifier for the api")
    local_time_serie: LocalTimeSerie
    data_source_description: Optional[str] = Field(None, description="Descriptions of the data source")
    data_frequency_id: str = DataFrequency
    assets_in_data_source:Optional[List[int]]

    def __str__(self):
        return self.class_name() +f"{self.unique_identifier}"

    def append_assets(self, asset_id_list:list, timeout=None):
        url = f"{self.get_object_url()}/{self.id}/append_assets/"

        payload = {"json": {"asset_id_list":asset_id_list}}
        r = make_request(s=self.build_session(), loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload,
                         time_out=timeout)
        if r.status_code in [200] == False:
            raise Exception(f" {r.text()}")
        return self.__class__(**r.json())

class HistoricalBarsSource(TDAGAPIDataSource):
    execution_venues: list
    data_mode: Literal['live', 'backtest'] = Field(
        description="Indicates whether the source is for live data or backtesting."
    )
    adjusted:bool