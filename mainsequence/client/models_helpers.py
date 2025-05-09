from .models_vam import *
from .base import MARKETS_CONSTANTS
from .models_tdag import LocalTimeSerie

from pydantic import  ConfigDict


def get_right_account_class(account: Account):
    from mainsequence.client import models_vam as model_module
    execution_venue_symbol = account.execution_venue.symbol
    AccountClass = getattr(model_module, MARKETS_CONSTANTS.ACCOUNT_VENUE_FACTORY[execution_venue_symbol])
    account, _ = AccountClass.get(id=account.id)
    return account


class MarketsTimeSeriesDetails(BaseObjectOrm, BasePydanticModel):
    id:Optional[int]=None
    unique_identifier: str
    related_local_time_serie: Union[LocalTimeSerie,int]
    description: Optional[str] = Field(None, description="Descriptions of the data source")
    data_frequency_id: str = DataFrequency
    assets_in_data_source:Optional[List[int]]
    extra_properties: Optional[Dict]

    def __str__(self):
        return self.class_name() + f" {self.unique_identifier}"

    @classmethod
    def get(cls,*args,**kwargs):
        return super().get(*args,**kwargs)

    @classmethod
    def filter(cls,*args,**kwargs):
        return super().filter(*args,**kwargs)

    def append_asset_list_source(self, asset_list: List[Asset]):
        if asset_list:
            asset_id_list = [a.id for a in asset_list]
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
        asset_list:List[Asset],
        description=""
    ):

        # if run for the first time save this as reference in VAM
        bar_source = MarketsTimeSeriesDetails.update_or_create(
            unique_identifier=unique_identifier,
            related_local_time_serie__id=time_serie.local_time_serie.id,
            data_frequency_id=data_frequency_id,
            description=description,
        )

        if bar_source is None:
            raise ValueError("No historical bars source found")

        bar_source.append_asset_list_source(asset_list=asset_list)


class  AccountValuationTSDetails(BaseObjectOrm, BasePydanticModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    markets_time_serie_detail: int
    execution_venue: int
    valuation_column_name: str
    missing_price_assets:List[int]

    @classmethod
    def create_or_update(cls,markets_time_serie_detail__id:int,execution_venue__symbol:str,
                         valuation_column_name:str,
                         timeout=None):
        url = f"{cls.get_object_url()}/create_or_update/"

        payload = {"json": {"markets_time_serie_detail__id": markets_time_serie_detail__id,
                            "execution_venue__symbol":execution_venue__symbol,
                            "valuation_column_name":valuation_column_name
                            }}
        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="POST", url=url, payload=payload,
                         time_out=timeout)
        if r.status_code in [200,201] == False:
            raise Exception(f" {r.text()}")
        return cls(**r.json())