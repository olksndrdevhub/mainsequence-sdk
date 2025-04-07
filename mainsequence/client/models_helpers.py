from .models_vam import *
from .base import MARKETS_CONSTANTS
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
    AccountClass = getattr(model_module, MARKETS_CONSTANTS.ACCOUNT_VENUE_FACTORY[execution_venue_symbol])
    account, _ = AccountClass.get(id=account.id)
    return account


class MarketsTimeSeriesDetails(BaseObjectOrm, BasePydanticModel):
    id:Optional[int]=None
    unique_identifier: str
    related_local_time_serie: LocalTimeSerie
    description: Optional[str] = Field(None, description="Descriptions of the data source")
    data_frequency_id: str = DataFrequency
    assets_in_data_source:Optional[List[int]]
    extra_properties: Optional[Dict]

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
        description=""
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
                    description=description,
                )

        if bar_source is None:
            raise ValueError("No historical bars source found")

        bar_source.append_asset_list_source(time_serie)


    def get_last_observation(self, execution_venue_symbol=None, timeout=None):

        # get the last observation
        last_observation_date = self.related_local_time_serie.remote_table.sourcetableconfiguration.get_data_updates().max_time_index_value
        last_obs = self.related_local_time_serie.get_data_between_dates_from_api(
            start_date=last_observation_date,
            end_date=None,
            great_or_equal=True,
            less_or_equal=False,
            unique_identifier_list=None,
            columns=None,
            unique_identifier_range_map=None
        )

        if execution_venue_symbol is None:
            return last_obs

        # get the asset type for the venue
        VENUE_MAP = {}
        for v in CONSTANTS.BINANCE_VENUES:
            VENUE_MAP[v] = CONSTANTS.ASSET_TYPE_CRYPTO_SPOT
        VENUE_MAP[CONSTANTS.ALPACA_EV_SYMBOL] = CONSTANTS.ASSET_TYPE_CASH_EQUITY
        asset_type = VENUE_MAP[execution_venue_symbol]

        # parse the combined unique symbol
        last_obs[['symbol', 'asset_type']] = last_obs['unique_identifier'].str.split('-*-', expand=True).drop(columns=[1])
        last_obs = last_obs[last_obs["asset_type"] == asset_type]
        asset_list = Asset.filter(asset_type=asset_type, execution_venue__symbol=execution_venue_symbol, symbol__in=last_obs["symbol"].to_list())
        last_obs["unique_identifier"] = last_obs["symbol"].map({a.symbol: a.unique_identifier for a in asset_list})
        return last_obs



class HistoricalBarsSource(MarketsTimeSeriesDetails):
    execution_venues: list
    data_mode: Literal['live', 'backtest'] = Field(
        description="Indicates whether the source is for live data or backtesting."
    )
    adjusted:bool

    @classmethod
    def register_in_backend(
            cls,
            unique_identifier:str,
            time_serie,
            execution_venues_symbol,
            data_mode,
            description: str = "",
            create_bars: bool = True
    ):
        bar_source = None
        try:
            bar_source = cls.get(
                data_frequency_id=time_serie.frequency_id,
                execution_venues__symbol__in=[execution_venues_symbol],
                data_mode=data_mode
            )

            bar_source = bar_source.patch(related_local_time_serie__id=time_serie.local_time_serie.id)

        except Exception as e:
            print(f"Exception when getting historical bar source {e}")

            # if run for the first time save this as reference in VAM
            bar_source = cls.update_or_create(
                unique_identifier=f"{execution_venues_symbol}_{time_serie.frequency_id}",
                related_local_time_serie__id=time_serie.local_time_serie.id,
                description=description,
                execution_venues_symbol__in=[execution_venues_symbol],
                data_frequency_id=time_serie.frequency_id,
                data_mode=data_mode
            )

        if bar_source is None:
            raise ValueError("No historical bars source found")

        bar_source.append_asset_list_source(time_serie)