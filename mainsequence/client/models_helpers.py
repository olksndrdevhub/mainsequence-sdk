from .models_vam import *
from .base import VAM_CONSTANTS
from .models_tdag import LocalTimeSerie

def get_model_class(model_class: str):
    """
    Reverse look from model class by name
    """
    MODEL_CLASS_MAP = {
        
        "Asset": Asset,
        "AssetCurrencyPair": AssetCurrencyPair,
        "AssetFutureUSDM": AssetFutureUSDM,
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


class MarketsTimeSeriesDetails(BaseObjectOrm, BasePydanticModel):
    id: int
    unique_identifier: str
    related_local_time_serie: LocalTimeSerie
    data_source_description: Optional[str] = Field(None, description="Descriptions of the data source")
    data_frequency_id: str = DataFrequency
    assets_in_data_source:Optional[List[int]]

    def __str__(self):
        return self.class_name() + f" {self.unique_identifier}"

    def append_asset_list_source(self, time_serie: "TimeSerie"):
        if time_serie.asset_list:
            asset_id_list = [a.id for a in time_serie.asset_list]
            self.append_assets(asset_id_list=asset_id_list)
            print("Added assets to bars")

    def append_assets(self, asset_id_list:list, timeout=None):
        url = f"{self.get_object_url()}/{self.id}/append_assets/"

        payload = {"json": {"asset_id_list":asset_id_list}}
        r = make_request(s=self.build_session(), loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload,
                         time_out=timeout)
        if r.status_code in [200] == False:
            raise Exception(f" {r.text()}")
        return self.__class__(**r.json())

    @classmethod
    def register_in_backend(
        cls,
        unique_identifier,
        time_serie,
        data_frequency_id,
        data_source_description=""
    ):
        try:
            bar_source = MarketsTimeSeriesDetails.get(
                unique_identifier=unique_identifier,
            )

            if time_serie.use_vam_assets and bar_source.related_local_time_serie.id != time_serie.local_time_serie.id:
                bar_source = bar_source.patch(related_local_time_serie__id=time_serie.local_time_serie.id)

        except DoesNotExist:
            if time_serie.use_vam_assets:
                # if run for the first time save this as reference in VAM
                bar_source = MarketsTimeSeriesDetails.update_or_create(
                    unique_identifier=unique_identifier,
                    related_local_time_serie__id=time_serie.local_time_serie.id,
                    data_frequency_id=data_frequency_id,
                    data_source_description=data_source_description,
                )

        if bar_source is None:
            raise ValueError("No historical bars source found")

        bar_source.append_asset_list_source(time_serie)

class HistoricalBarsSource(MarketsTimeSeriesDetails):
    execution_venues: list
    data_mode: Literal['live', 'backtest'] = Field(
        description="Indicates whether the source is for live data or backtesting."
    )
    adjusted:bool

    @classmethod
    def register_in_backend(
            cls,
            time_serie,
            execution_venues_symbol,
            data_mode,
            data_source_description: str = "",
            create_bars: bool = True
    ):
        bar_source = None
        try:
            bar_source = cls.get(
                data_frequency_id=time_serie.frequency_id,
                execution_venues__symbol__in=[execution_venues_symbol],
                data_mode=data_mode
            )
            if time_serie.use_vam_assets and bar_source.related_local_time_serie.id != time_serie.local_time_serie.id:
                bar_source = bar_source.patch(related_local_time_serie__id=time_serie.local_time_serie.id)

        except Exception as e:
            print(f"Exception when getting historical bar source {e}")
            if time_serie.use_vam_assets and create_bars:
                # if run for the first time save this as reference in VAM
                bar_source = cls.update_or_create(
                    unique_identifier=f"{execution_venues_symbol}_{time_serie.frequency_id}",
                    related_local_time_serie__id=time_serie.local_time_serie.id,
                    data_source_description=data_source_description,
                    execution_venues_symbol__in=[execution_venues_symbol],
                    data_frequency_id=time_serie.frequency_id,
                    data_mode=data_mode
                )

        if bar_source is None:
            raise ValueError("No historical bars source found")

        bar_source.append_asset_list_source(time_serie)