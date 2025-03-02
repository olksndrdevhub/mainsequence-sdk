import copy
import datetime
from multiprocessing.managers import BaseManager

import pytz
import requests
from functools import wraps
import pandas as pd
from typing import Union,Literal
from types import SimpleNamespace
import requests
import os
import json
import time

from enum import IntEnum, Enum



from .utils import AuthLoaders, make_request, DoesNotExist, request_to_datetime, CONSTANTS, VAM_API_ENDPOINT, \
    VAM_ENDPOINT
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator,root_validator
import time
import inspect

loaders = AuthLoaders()

DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


class HtmlSaveException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        self.file_path = None

        if 'html' in message.lower():
            self.file_path = self.save_as_html_file()

    def save_as_html_file(self):
        # Get the name of the method that raised the exception
        caller_method = inspect.stack()[2].function

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create the directory to save HTML files if it doesn't exist
        folder_path = 'html_exceptions'
        os.makedirs(folder_path, exist_ok=True)

        # Create the filename
        filename = f"{caller_method}_{timestamp}.html"
        file_path = os.path.join(folder_path, filename)

        # Save the message as an HTML file
        with open(file_path, 'w') as file:
            file.write(self.message)

        return file_path

    def __str__(self):
        if self.file_path:
            return f"HTML content saved to {self.file_path}"
        else:
            return self.message


def validator_for_string(value):
    if isinstance(value, str):
        # Parse the string to a datetime object
        try:
            return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            raise ValueError(f"Invalid datetime format: {value}. Expected format is 'YYYY-MM-DDTHH:MM:SSZ'.")



DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


class BaseVamPydanticModel(BaseModel):
    orm_class: str = None  # This will be set to the class that inherits

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Set orm_class to the class itself
        cls.orm_class = cls.__name__



    def to_serialized_dict(self):
        new_dict=json.loads(self.model_dump_json())
        if hasattr(self,'unique_identifier'):
            new_dict['unique_identifier'] = self.unique_identifier

        return new_dict


def get_correct_asset_class(asset_type):
    if asset_type in [CONSTANTS.ASSET_TYPE_CRYPTO_SPOT, CONSTANTS.ASSET_TYPE_CASH_EQUITY,CONSTANTS.ASSET_TYPE_CURRENCY]:
        return Asset
    elif asset_type in [CONSTANTS.ASSET_TYPE_CRYPTO_USDM]:
        return AssetFutureUSDM
    elif asset_type in [CONSTANTS.ASSET_TYPE_CURRENCY_PAIR]:
        return CurrencyPair
    else:
        raise NotImplementedError

def resolve_asset(asset_dict:dict):

    AssetClass=get_correct_asset_class(asset_dict['asset_type'])
    asset=AssetClass(**asset_dict)

    return asset
def build_session(loaders):
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    s.headers.update(loaders.auth_headers)


    retries = Retry(total=2, backoff_factor=2, )
    s.mount('http://', HTTPAdapter(max_retries=retries))
    return s

session=build_session(loaders=loaders)

class BaseObjectOrm:
    END_POINTS = {
        "User":"user",
        "TargetPortfolio": 'target_portfolio',

        "Asset": "asset",
        "IndexAsset": "index_asset",
        "AssetFutureUSDM": "asset_future_usdm",
        "VirtualFund": "virtualfund",
        "OrderManager": "order_manager",
        "ExecutionVenue": "execution_venue",
        "Order": "order",
        "OrderEvent": "order_event",
        "Account": "account",
        "Trade": "trade",
        "VirtualFundHistoricalHoldings": "historical_holdings",
        "AccountHistoricalHoldings": "account_historical_holdings",
        "AccountRiskFactors":"account_risk_factors",
        "AccountPortfolioScheduledRebalance": "account_portfolio_scheduled_rebalance",
        "AccountPortfolioHistoricalPositions":"account_portfolio_historical_positions",
        "ExecutionPrediction": "execution_predictions",
        "ExecutionPositions": "execution_positions",
        "AccountCoolDown": "account_cooldown",
        "HistoricalWeights": "portfolio_weights",
        "TargetPortfolioIndexAsset":"target_portfolio_index_asset",

        "HistoricalBarsSource":"data_sources/historical-bars-source",
        "TDAGAPIDataSource": "data_sources/tdag-api-data-source",
        "AssetCategory":"asset-category",
        "TargetPortfolioFrontEndDetails": "target-portfolio-details",

    }
    ROOT_URL = VAM_API_ENDPOINT
    LOADERS = loaders


    @staticmethod
    def request_to_datetime(string_date: str):
        return request_to_datetime(string_date=string_date)
    
    @staticmethod
    def date_to_string(target_date:datetime.datetime):
        return target_date.strftime(DATE_FORMAT)


    @classmethod
    def class_name(cls):
        if hasattr(cls,"CLASS_NAME"):
            return cls.CLASS_NAME
        return cls.__name__


    @classmethod
    def build_session(cls):
        s=session
        return s

    @property
    def s(self):
        s = self.build_session()
        return s

    def ___hash__(self):

        if hasattr(self, "unique_identifier"):
            return self.unique_identifier

        return self.id
    def __repr__(self):
        object_id=self.id if hasattr(self, "id") else None
        return f"{self.class_name()}: {object_id}"

    @classmethod
    def get_object_url(cls):
        url=f"{cls.ROOT_URL}/{cls.END_POINTS[cls.class_name()]}"
        return  url
    
    @staticmethod
    def _parse_parameters_filter(parameters):
        for key, value in parameters.items():
            if "__in" in key:
                assert isinstance(value, list)
                value = [str(v) for v in value]
                parameters[key] = ",".join(value)
        return parameters

    @classmethod
    def filter(cls, timeout=None, **kwargs):
        """
        Fetches *all pages* from a DRF-paginated endpoint.
        Accumulates results from each page until 'next' is None.

        Returns a list of `cls` objects (not just one page).

        DRF's typical paginated response looks like:
            {
              "count": <int>,
              "next": <str or null>,
              "previous": <str or null>,
              "results": [ ...items... ]
            }
        """
        base_url = cls.get_object_url()  # e.g. "https://api.example.com/assets"
        params = cls._parse_parameters_filter(kwargs)

        # We'll handle pagination by following the 'next' links from DRF.
        accumulated = []
        next_url = f"{base_url}/"  # Start with the main endpoint (list)

        while next_url:
            # For each page, do a GET request
            r = make_request(
                s=cls.build_session(),
                loaders=cls.LOADERS,
                r_type="GET",
                url=next_url,  # next_url changes each iteration
                payload={"params": params},
                timeout=timeout
            )

            if r.status_code != 200:
                # Handle errors or break out
                if r.status_code == 401:
                    raise Exception("Unauthorized. Please add credentials to environment.")
                elif r.status_code == 500:
                    raise Exception("Server Error.")
                else:
                    raise Exception(r.status_code)

            data = r.json()
            # data should be a dict with "count", "next", "previous", and "results".

            # DRF returns the next page URL in `data["next"]`
            next_url = data["next"]  # either a URL string or None

            # data["results"] should be a list of objects
            for item in data["results"]:
                # Insert "orm_class" if you still need that
                item["orm_class"] = cls.__name__
                try:
                    accumulated.append(cls(**item) if issubclass(cls,BaseVamPydanticModel) else item)
                except Exception as e:
                    raise e


            # We set `params = None` (or empty) after the first loop to avoid appending repeatedly
            # but only if DRF's `next` doesn't contain the query parameters.
            # Usually, DRF includes them, so you don't need to do anything special here.
            params = None

        return accumulated


    @classmethod
    def get(cls, pk, timeout=None):
        """
        Retrieves exactly one object by primary key: GET /base_url/<pk>/
        Raises `DoesNotExist` if 404 or the response is empty.
        Raises Exception if multiple or unexpected data is returned.
        """
        base_url = cls.get_object_url()
        url = f"{base_url}/{pk}/"  # e.g. https://api.example.com/assets/123/

        r = make_request(
            s=cls.build_session(),
            loaders=cls.LOADERS,
            r_type="GET",
            url=url,
            payload={},  # Typically no query params for a single retrieve
            timeout=timeout
        )

        if r.status_code == 404:
            raise DoesNotExist(f"No object found for pk={pk}")
        elif r.status_code == 401:
            raise Exception("Unauthorized. Please add credentials to environment.")
        elif r.status_code == 500:
            raise Exception("Server Error")
        elif r.status_code != 200:
            # Some other error; handle as appropriate
            raise Exception(f"Unexpected status code: {r.status_code}")

        # Expecting a single JSON object
        data = r.json()
        # Instantiate the single object
        data["orm_class"] = cls.__name__
        return cls(**data)

    @staticmethod
    def serialize_for_json(kwargs):
        new_data={}
        for key,value in kwargs.items():
            new_value=copy.deepcopy(value)
            if isinstance(value,datetime.datetime):
                new_value=str(value)

            new_data[key]=new_value
        return new_data
    @classmethod
    def create(cls,timeout=None,*args,**kwargs):
        """

        :return:
        :rtype:
        """
        base_url = cls.get_object_url()
        data =cls.serialize_for_json(kwargs)
        payload={"json":data}

        r = make_request(s=cls.build_session(),loaders=cls.LOADERS, r_type="POST", url=f"{base_url}/", payload=payload,
                         timeout=timeout
                         )
        if r.status_code not in [201]:
           raise Exception(r.text)
        return cls(** r.json())

    @classmethod
    def update_or_create(cls, timeout=None, *args, **kwargs):
        """

        :return:
        :rtype:
        """
        url = f"{cls.get_object_url()}/update_or_create/"
        data = cls.serialize_for_json(kwargs)
        payload={"json":data}

        r = make_request(s=cls.build_session(),loaders=cls.LOADERS, r_type="POST", url=url, payload=payload,
                         timeout=timeout
                         )
        if r.status_code not in [201, 200]:
           raise Exception(r.text)
        return cls(** r.json())
    
    @classmethod
    def destroy_by_id(cls,instance_id,*args,**kwargs):
        base_url = cls.get_object_url()
        data = cls.serialize_for_json(kwargs)
        payload = {"json": data}
        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="DELETE", url=f"{base_url}/{instance_id}/",
                         payload=payload)
        if r.status_code != 204:
            raise Exception(r.text)

    @classmethod
    def patch_by_id(cls,instance_id, *args, **kwargs):
        """

        :return:
        :rtype:
        """
        base_url = cls.get_object_url()
        data = cls.serialize_for_json(kwargs)
        payload = {"json": data}
        r = make_request(s=cls.build_session(),loaders=cls.LOADERS, r_type="PATCH", url=f"{base_url}/{instance_id}/", payload=payload)
        if r.status_code != 200:
            raise HtmlSaveException(r.text)
        return cls(**r.json())

    def patch(self,*args,**kwargs):
        return self.__class__.patch_by_id(self.id,*args,**kwargs)
    
    def delete(self,*args,**kwargs):
        return self.__class__.destroy_by_id(self.id)




class ExecutionPositions(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int] = None
    positions:List["WeightExecutionPosition"]
    target_portfolio:Union["TargetPortfolio",int]
    positions_date:datetime.datetime
    comments:Optional[str]=None
    received_in_execution_engine: bool
    execution_configuration:"TargetPortfolioExecutionConfiguration"

    @property
    def symbol_asset_map(self):
        return {p.asset.unique_symbol: p.asset.id for p in self.positions}

    @property
    def symbol_to_id_map(self):
        return {p.asset.id: Asset(**p.asset.model_dump()) for p in self.positions}

    @property
    def broker_config(self):
        return self.execution_configuration.orders_execution_configuration.broker_config

    @property
    def broker_class(self):
        return self.execution_configuration.orders_execution_configuration.broker_class

    @classmethod
    def add_from_time_serie(cls, time_serie_signal_hash_id: str, positions_list: list,
                            positions_time: datetime.datetime,
                            comments: Union[str, None] = None, timeout=None):
        """

        :param session:
        :return:
        """
        url = f"{cls.get_object_url()}/add_from_time_serie/"
        payload = {"json": {"time_serie_signal_hash_id": time_serie_signal_hash_id,
                            "positions_time": positions_time.strftime(DATE_FORMAT),
                            "positions_list": positions_list,

                            }, }

        r = make_request(s=cls.build_session(),
                         loaders=cls.LOADERS, r_type="POST", url=url, payload=payload, timeout=timeout)
        if r.status_code not in [201, 204] :
            raise HtmlSaveException(r.text)
        return [cls(**e) for e in r.json()]


class WeightExecutionPosition(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int] = None
    parent_execution_positions: int
    asset: Union["Asset", "AssetFutureUSDM",int]
    weight_notional_exposure: float

    @property
    def asset_id(self):
        return self.asset if isinstance(self.asset,int) else self.asset.id

    @root_validator(pre=True)
    def resolve_assets(cls, values):
        # Check if 'asset' is a dict and determine its type
        if isinstance(values.get('asset'), dict):
            asset = values.get('asset')
            asset = resolve_asset(asset_dict=asset)
            values['asset'] = asset

        return values

class AccountCoolDown(BaseObjectOrm):
    
    pass
class Calendar(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int] = None
    name: str
    calendar_dates:Optional[dict]=None



class Organization(BaseModel):
    id: int
    uid: str
    name: str
    url: Optional[str]  # URL can be None

class Group(BaseModel):
    id: int
    name: str
    permissions: List[Any]  # Adjust the type for permissions as needed

class User(BaseObjectOrm,BaseVamPydanticModel):
    id: int
    password: str
    is_superuser: bool
    first_name: str
    last_name: str
    is_staff: bool
    is_active: bool
    date_joined: datetime.datetime
    role: str
    username: str
    email: str
    last_login: datetime.datetime
    api_request_limit: int
    mfa_secret: Optional[str]
    mfa_enabled: bool
    organization: Organization
    plan: Optional[Any]  # Use a specific model if plan details are available
    groups: List[Group]
    user_permissions: List[Any]  # Adjust as necessary for permission structure

    @classmethod
    def get_object_url(cls):
        url = f"{cls.ROOT_URL.replace('orm/api', 'users/api')}/{cls.END_POINTS[cls.class_name()]}"
        return url
    @classmethod
    def get_authenticated_user_details(cls):
        url = f"{cls.get_object_url()}/get_user_details/"

        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="GET", url=url,)
        if r.status_code not in [200, 201]:
            raise Exception(f" {r.text()}")

        return cls(**r.json())


class AssetMixin(BaseObjectOrm, BaseVamPydanticModel):

    id: Optional[int] = None
    symbol: str
    name: str
    asset_type: str
    can_trade: bool
    calendar: Union[Calendar,int]
    execution_venue: Union["ExecutionVenue", int]
    delisted_datetime: Optional[datetime.datetime] = None
    unique_identifier: str
    unique_symbol: Optional[str]=None

    @staticmethod
    def get_properties_from_unique_symbol(unique_symbol: str):
        return {"symbol": unique_symbol}

    @property
    def execution_venue_symbol(self):
        return self.execution_venue.symbol

    @classmethod
    def switch_cash_asset(cls,asset_id,target_currency_asset:object):
        url = f"{cls.get_object_url()}/{asset_id}/switch_cash_asset"
        payload=dict(params={"target_currency_asset_id":target_currency_asset.id
                                      })
        r = make_request(s=cls.build_session(),loaders=cls.LOADERS, r_type="GET", url=url, payload=payload)

        if r.status_code != 200:
            raise Exception("Error switching cash asset")
        return cls(**r.json())

    @classmethod
    def switch_cash_in_asset_list(cls, asset_id_list:list, target_currency_asset_id: int,timeout=None):
        url = f"{cls.get_object_url()}/switch_cash_in_asset_list/"
        payload = dict(json={"asset_id_list": asset_id_list,"target_currency_asset_id":target_currency_asset_id
                               })
        r = make_request(s=cls.build_session(), loaders=cls.LOADERS,r_type="POST", url=url, payload=payload,
                         timeout=timeout)

        if r.status_code != 200:
            raise Exception("Error switching cash asset")
        return r.json()

    @classmethod
    def batch_upsert(cls,asset_config_list:list, execution_venue_symbol:str,asset_type:str,timeout=None):
        """

        Parameters
        ----------
        asset_config_list
        execution_venue_symbol
        asset_type

        Returns
        -------

        """

        url = f"{cls.get_object_url()}/batch_upsert/"
        payload = dict(json={"asset_config_list": asset_config_list,"execution_venue_symbol":execution_venue_symbol,
                             "asset_type":asset_type
                               })
        r = make_request(s=cls.build_session(),loaders=cls.LOADERS, r_type="POST", url=url, timeout=timeout,
                         payload=payload)

        if r.status_code != 200:
            raise Exception("Error inserting assets")
        return r.json()


    def get_ccxt_symbol(self,settlement_symbol:Union[str,None]=None):
        """
        Gets the right symbol for ccxt
        Returns
        -------

        """
        if self.asset_type in [CONSTANTS.ASSET_TYPE_CRYPTO_USDM]:
            return  f"{self.base_asset.symbol}/{self.quote_asset.symbol}:{self.quote_asset.symbol}"
        else:
            return  f"{self.symbol}/{settlement_symbol}:{settlement_symbol}"

    @classmethod
    def get_all_assets_on_positions(cls, execution_venue_symbol:str, asset_type: str,
                                    switch_to_currency_pair_with_quote:Union[str,None]=None,):
        url = f"{Asset.get_object_url()}/get_all_assets_on_positions"
        payload = dict(params={"execution_venue_symbol": execution_venue_symbol,
                               "asset_type":asset_type,

                               })
        if switch_to_currency_pair_with_quote is not None:
            payload["params"]["switch_to_currency_pair_with_quote"]=switch_to_currency_pair_with_quote
        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="GET", url=url, payload=payload)

        if r.status_code != 200:
            raise Exception("Error getting all assets in positions")
        return [cls(**a) for a in r.json()]

    @classmethod
    def filter_with_asset_class(cls, timeout=None, *args, **kwargs):
        """
        Filters assets and returns instances with their correct asset class,
        looping through all DRF-paginated pages.
        """
        from .models_helpers import create_from_serializer_with_class

        base_url = cls.get_object_url()
        # Convert `kwargs` to query parameters
        params = cls._parse_parameters_filter(parameters=kwargs)

        # We'll call the custom action endpoint
        url = f"{base_url}/list_with_asset_class/"
        all_results = []

        # Build a single requests session
        s = cls.build_session()

        while url:
            # Make the request to the current page URL
            request_kwargs = {"params": params} if params else {}
            r = make_request(
                s=s,
                loaders=cls.LOADERS,
                r_type="GET",
                url=url,
                payload=request_kwargs,
                timeout=timeout
            )

            if r.status_code != 200:
                raise Exception(f"Error getting assets (status code: {r.status_code})")

            data = r.json()

            # Check if it's a DRF paginated response by looking for "results"
            if isinstance(data, dict) and "results" in data:
                # Paginated response
                results = data["results"]
                next_url = data["next"]
            else:
                # Either not paginated or no "results" key
                # It's possible your endpoint returns a plain list or other structure
                # Adjust accordingly if needed
                results = data
                next_url = None

            # Accumulate the results
            all_results.extend(results)

            # Prepare for the next loop iteration
            url = next_url
            # After the first request, DRF's `next` link is a full URL that already includes
            # appropriate query params, so we set `params=None` to avoid conflicts.
            params = None

        # Convert the accumulated raw data into asset instances with correct classes
        return create_from_serializer_with_class(all_results)
        
class AssetCategory(BaseObjectOrm, BaseVamPydanticModel):
    id:int
    unique_id:str
    name:str
    source:str
    assets:List[int]
    organization_owner_uid:str
    
    def __repr__(self):
        return self.name+" source:"+self.source

    def append_assets(self, asset_ids: List[int]) -> "AssetCategory":
        """
        Append the given asset IDs to this category.
        Expects a payload: {"assets": [<asset_id1>, <asset_id2>, ...]}
        """
        url = f"{self.get_object_url()}/{self.id}/append-assets/"
        payload = {"assets": asset_ids}
        r = make_request(
            s=self.build_session(),
            loaders=self.LOADERS,
            r_type="POST",
            url=url,
            payload={"json":payload}
        )
        if r.status_code not in [200, 201]:
            raise Exception(f"Error appending assets: {r.text()}")
        # Return a new instance of AssetCategory built from the response JSON.
        return AssetCategory(**r.json())

    def remove_assets(self, asset_ids: List[int]) -> "AssetCategory":
        """
        Remove the given asset IDs from this category.
        Expects a payload: {"assets": [<asset_id1>, <asset_id2>, ...]}
        """
        url = f"{self.get_object_url()}/{self.id}/remove-assets/"
        payload = {"assets": asset_ids}
        r = make_request(
            s=self.build_session(),
            loaders=self.LOADERS,
            r_type="POST",
            url=url,
            payload={"json": payload}
        )
        if r.status_code not in [200, 201]:
            raise Exception(f"Error removing assets: {r.text()}")
        # Return a new instance of AssetCategory built from the response JSON.
        return AssetCategory(**r.json())
    
class Asset(AssetMixin, BaseObjectOrm):

    def get_spot_reference_asset_symbol(self):
        """"
        """
        return self.symbol
    
    @classmethod
    def create_or_update_index_asset_from_portfolios(cls,
                                                     live_portfolio:int,
                                                     backtest_portfolio:int,
                                                     valuation_asset:int,
                                                     calendar:str,timeout=None
                                                     )->"TargetPortfolioIndexAsset":
        url = f"{cls.get_object_url()}/create_or_update_index_asset_from_portfolios/"
        payload = {"json": dict(  live_portfolio=live_portfolio,
                                                     backtest_portfolio=backtest_portfolio,
                                                     valuation_asset=valuation_asset,
                                                     calendar=calendar)}
        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="POST", url=url, payload=payload,
                         timeout=timeout)
        if r.status_code not in [200,201]:
            raise Exception(f" {r.text()}")

        return TargetPortfolioIndexAsset(**r.json())


class IndexAsset(Asset):
    valuation_asset: AssetMixin

class TargetPortfolioIndexAsset(IndexAsset):
    asset_type: str = CONSTANTS.ASSET_TYPE_INDEX
    can_trade:bool=False
    live_portfolio : "TargetPortfolio"
    backtest_portfolio : "TargetPortfolio"
    execution_venue: "ExecutionVenue"= Field(
        default_factory=lambda: ExecutionVenue(**CONSTANTS.VENUE_MAIN_SEQUENCE_PORTFOLIOS)
    )

    @property
    def live_portfolio_details_url(self):

        return f"{VAM_ENDPOINT}/dashboards/portfolio-detail/?target_portfolio_id={self.live_portfolio.id}"

    @property
    def backtest_portfolio_details_url(self):

        return f"{VAM_ENDPOINT}/dashboards/portfolio-detail/?target_portfolio_id={self.backtest_portfolio.id}"

class CurrencyPairMixin(AssetMixin, BaseVamPydanticModel):
    base_asset: Union[AssetMixin, int]
    quote_asset: Union[AssetMixin, int]

    def get_spot_reference_asset_symbol(self):

        base_asset_symbol = self.symbol.replace(self.quote_asset.symbol, "")
        return base_asset_symbol


class CurrencyPair(CurrencyPairMixin):
  pass

class FutureUSDMMixin(AssetMixin, BaseVamPydanticModel):
    maturity_code: str = Field(..., max_length=50)
    last_trade_time: Optional[datetime.datetime] = None
    currency_pair:CurrencyPair

    def get_spot_reference_asset_symbol(self):
        FUTURE_TO_SPOT_MAP = {
            CONSTANTS.BINANCE_FUTURES_EV_SYMBOL: {"1000SHIB": "SHIB"},
        }

        base_asset_symbol = self.symbol.replace(self.currency_pair.quote_asset.symbol, "")
        spot = FUTURE_TO_SPOT_MAP[self.execution_venue_symbol].get(base_asset_symbol, base_asset_symbol)
        return spot

class AssetFutureUSDM(FutureUSDMMixin,BaseObjectOrm):
    pass


class AccountPortfolioScheduledRebalance(BaseObjectOrm):

    pass



class AccountExecutionConfiguration(BaseVamPydanticModel):
    related_account: int  # Assuming related_account is represented by its ID
    rebalance_tolerance_percent: float = Field(0.02, ge=0)
    minimum_notional_for_a_rebalance: float = Field(15.00, ge=0)
    max_latency_in_cdc_seconds: float = Field(60.00, ge=0)
    force_market_order_on_execution_remaining_balances: bool = Field(False)
    orders_execution_configuration: Dict[str, Any]
    cooldown_configuration: Dict[str, Any]


class AccountPortfolioPosition(BaseVamPydanticModel):
    id: Optional[int]
    parent_positions: int
    target_portfolio: int
    weight_notional_exposure: Optional[float]=0.0
    constant_notional_exposure: Optional[float]=0.0
    single_asset_quantity: Optional[float]=0.0
    assets_in_position: list["WeightPosition"]

class AccountPortfolioHistoricalPositions(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int]
    positions_date: datetime.datetime
    comments: Optional[str]
    positions: list[AccountPortfolioPosition]
  

class AccountTargetPortfolio(BaseObjectOrm,BaseVamPydanticModel):
    related_account:Optional[int]
    latest_positions:Optional[AccountPortfolioHistoricalPositions]=None
    @property
    def unique_identifier(self):
        return self.related_account_id


class AccountMixin(BaseVamPydanticModel):
    id: Optional[int] = None
    execution_venue: "ExecutionVenue"
    account_is_active: bool
    account_name: Optional[str] = None
    is_account_in_cool_down:bool
    cash_asset: Asset
    is_paper:bool
    is_on_manual_rebalance: bool
    user: Optional[int] = None
    execution_configuration:"AccountExecutionConfiguration"
    account_target_portfolio: AccountTargetPortfolio
    latest_holdings:Union["AccountLatestHoldingsSerializer",None]=None


    @property
    def account_target_portfolio(self):
        return self.accounttargetportfolio


    @property
    def execution_venue_symbol(self):
        return self.execution_venue.symbol
    
    
    
    

    def get_latest_income_record(self):
        base_url = self.get_object_url()
        url = f"{base_url}/{self.id}/last_income_record/"
        r = make_request(s=self.build_session(), loaders=self.LOADERS, r_type="GET", url=url)
        if r.status_code != 200:
            raise Exception("Error Syncing funds in account")
        return FundingFeeTransaction(**r.json())

    def get_nav(self):
        base_url = self.get_object_url()
        url = f"{base_url}/{self.id}/get_nav"
        r = make_request(s=self.build_session(), loaders=self.LOADERS, r_type="GET", url=url)
        if r.status_code != 200:
            raise Exception(f"Error Getting NAV in account {r.text}")
        return r.json()

    def build_rebalance(self, latest_holdings: "AccountHistoricalHoldings",
                        tolerance: float,
                        change_cash_asset_to_currency_asset: Union[Asset, None] = None,
                        ):
        """

        :param latest_holdings:
        :type latest_holdings:
        :param tolerance:
        :type tolerance:
        :param change_cash_asset_to_currency_asset:
        :type change_cash_asset_to_currency_asset:
        :return:

         rebalance[<ASSET_ID>] = {"rebalance":{"quantity":, "reference_price":, "reference_notional":}, "asset":Asset}

        :rtype:
        """

        nav = self.get_nav()
        nav, nav_date = nav["nav"], nav["nav_date"]
        related_expected_asset_exposure_df = latest_holdings.related_expected_asset_exposure_df
        # extract Target Rebalance

        # extract expected holdings
        try:
            implicit_holdings_df = related_expected_asset_exposure_df.groupby("aid") \
                .aggregate({"holding": "sum", "price": "last", "expected_holding_in_fund": "sum"}) \
                .rename(columns={"expected_holding_in_fund": "expected_holding"})
        except Exception as e:
            raise e
        implicit_holdings_df["difference"] = (
                    implicit_holdings_df["expected_holding"] - implicit_holdings_df["holding"])
        implicit_holdings_df["relative_w"] = (implicit_holdings_df["difference"] * implicit_holdings_df["price"]) / nav
        implicit_holdings_df["tolerance_flag"] = implicit_holdings_df["relative_w"].apply(
            lambda x: 1 if x >= tolerance else 0)
        implicit_holdings_df["difference"] = implicit_holdings_df["difference"] * implicit_holdings_df[
            "tolerance_flag"]
        implicit_holdings_df["expected_holding"] = implicit_holdings_df["holding"] + implicit_holdings_df[
            "difference"]

        implicit_holdings = implicit_holdings_df[["expected_holding", "price"]] \
            .rename(columns={"expected_holding": "holding"}).T.to_dict()

        implicit_holdings_df["reference_notional"] = implicit_holdings_df["price"] * implicit_holdings_df["difference"]
        rebalance = implicit_holdings_df[["difference", "reference_notional", "price"]] \
            .rename(columns={"difference": "quantity", "price": "reference_price"}).T.to_dict()

        all_assets = implicit_holdings.keys()
        new_rebalance, new_implicit_holdings = {}, {}
        # build_asset_switch
        asset_switch_map = Asset.switch_cash_in_asset_list(
            asset_id_list=[c for c in all_assets if c != change_cash_asset_to_currency_asset.id],
            target_currency_asset_id=int(change_cash_asset_to_currency_asset.id))
        asset_switch_map[
            str(change_cash_asset_to_currency_asset.id)] = change_cash_asset_to_currency_asset.serialized_config

        for a_id in all_assets:
            try:
                new_a = Asset(**asset_switch_map[str(a_id)])
            except Exception as e:
                raise e
            if rebalance[a_id]["quantity"] != 0.0:
                new_rebalance[new_a.id] = {"rebalance": rebalance[a_id],
                                               "asset": new_a}
            try:
                new_implicit_holdings[new_a.id] = implicit_holdings[a_id]
            except Exception as e:
                raise e
        not_rebalanced_by_tolerance = implicit_holdings_df[implicit_holdings_df["difference"] != 0]
        not_rebalanced_by_tolerance = not_rebalanced_by_tolerance[not_rebalanced_by_tolerance["tolerance_flag"] == 0][
            "relative_w"]
        not_rebalanced_by_tolerance = {"tolerance": not_rebalanced_by_tolerance.to_dict()}
        return new_rebalance, new_implicit_holdings, not_rebalanced_by_tolerance


    def get_latest_holdings(self):
        base_url = self.get_object_url()
        url = f"{base_url}/{self.id}/latest_holdings/"
        r = make_request(s=self.build_session(),loaders=self.LOADERS, r_type="GET", url=url)
        if r.status_code != 200:
            raise Exception("Error Syncing funds in account")
        return AccountHistoricalHoldings(**r.json())
    
    def get_missing_assets_in_exposure(self,asset_list_ids,asset_type:str,timeout=None)->list[Asset]:
        base_url = self.get_object_url()
        url = f"{base_url}/{self.id}/get_missing_assets_in_exposure/"
        payload = {"json": {"asset_list_ids":asset_list_ids,"asset_type":asset_type}}
        
        r = make_request(s=self.build_session(),payload=payload, loaders=self.LOADERS, r_type="GET", url=url,timeout=timeout)
        if r.status_code != 200:
            raise Exception(r.text)
        
        asset_list=[]
        for a in r.json():
            asset_list.append(resolve_asset(a))
        
        return  asset_list

class Account(AccountMixin,BaseObjectOrm,BaseVamPydanticModel):

    ...


class AccountLatestHoldingsSerializer(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int] = Field(None, primary_key=True)
    holdings_date: datetime.datetime
    comments: Optional[str] = Field(None, max_length=150)
    nav: Optional[float] = None


    is_trade_snapshot: bool = Field(default=False)
    target_trade_time: Optional[datetime.datetime] = None
    related_expected_asset_exposure_df: Optional[Dict[str, Any]] = None

    holdings: list

class AccountRiskFactors(BaseObjectOrm):

    ...


class AccountHistoricalHoldings(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int] = Field(None, primary_key=True)
    holdings_date: datetime.datetime
    comments: Optional[str] = Field(None, max_length=150)
    nav: Optional[float] = None

    related_account: Union[int,"Account"]
    is_trade_snapshot: bool = Field(default=False)
    target_trade_time: Optional[datetime.datetime] = None
    related_expected_asset_exposure_df: Optional[Dict[str, Any]] = None

    holdings: list

    @classmethod
    def destroy_holdings_before_date(cls,target_date:datetime.datetime,
                                     keep_trade_snapshots:bool):
        base_url = cls.get_object_url()
        payload = {"json": {"target_date":target_date.strftime(DATE_FORMAT),
                            "keep_trade_snapshots":keep_trade_snapshots}}


        r = make_request(s=cls.build_session(),
                         loaders=cls.LOADERS,r_type="POST", url=f"{base_url}/destroy_holdings_before_date/", payload=payload)
        if r.status_code != 204:
            raise Exception(r.text)




class FundingFeeTransaction(BaseObjectOrm):
    pass

class AccountPortfolioHistoricalWeights(BaseObjectOrm):
    pass




class WeightPosition(BaseObjectOrm, BaseVamPydanticModel):
    id: Optional[int] = None
    parent_weights: int
    asset: Union[AssetMixin,int]
    weight_notional_exposure: float

    @property
    def asset_id(self):
        return self.asset if isinstance(self.asset,int) else self.asset.id

    @root_validator(pre=True)
    def resolve_assets(cls, values):
        # Check if 'asset' is a dict and determine its type
        if isinstance(values.get('asset'), dict):
            asset=values.get('asset')
            asset=resolve_asset(asset_dict=asset)
            values['asset']=asset
         
        return values

class HistoricalWeights(BaseObjectOrm,BaseVamPydanticModel):
    id: int
    weights_date: datetime.datetime
    comments: Optional[str] = None
    target_portfolio: int
    weights:Union[List[WeightPosition],List[int]]

    @classmethod
    def add_from_time_serie(cls, local_time_serie_id: int, positions_list: list,
                            weights_date: datetime.datetime,
                            comments: Union[str, None] = None, timeout=None):
        """

        :param session:
        :return:
        """
        url = f"{cls.get_object_url()}/add_from_time_serie/"
        payload = {"json": {"local_time_serie_id": local_time_serie_id,
                            "weights_date": weights_date.strftime(DATE_FORMAT),
                            "positions_list": positions_list,

                            }, }

        r = make_request(s=cls.build_session(),
                         loaders=cls.LOADERS, r_type="POST", url=url, payload=payload, timeout=timeout)
        if r.status_code not in [201, 200]:
            raise Exception(f"Error inserting new weights {r.text}")
     
        return cls(**r.json())


class ExecutionVenue(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int] = None
    symbol: str
    name: str
    @property
    def unique_identifier(self):
        return f"{self.symbol}"


class DataFrequency(str, Enum):
    one_m = "1m"
    five_m = "5m"
    one_d = "1d"
    one_w = "1w"
    one_month ="1mo"

class TDAGAPIDataSource(BaseObjectOrm, BaseVamPydanticModel):
    id:Optional[int]
    unique_identifier: str = Field(..., description="Unique identifier for the api")
    data_source_id: int = Field(..., description="Unique identifier for the data source")
    local_hash_id: str = Field(..., max_length=64, description="Local hash ID of the data source")
    data_source_description: Optional[str] = Field(None, description="Descriptions of the data source")
    data_frequency_id: str = DataFrequency
    assets_in_data_source:Optional[List[int]]

    def __str__(self):
        return self.class_name() +f"{self.unique_identifier}"


    def append_assets(self,asset_id_list:list,timeout=None):
        url = f"{self.get_object_url()}/{self.id}/append_assets/"

        payload = {"json": {"asset_id_list":asset_id_list}}
        r = make_request(s=self.build_session(), loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload,
                         timeout=timeout)
        if r.status_code in [200] == False:
            raise Exception(f" {r.text()}")
        return self.__class__(**r.json())


class HistoricalBarsSource(TDAGAPIDataSource):
    execution_venues: list
    data_mode:Literal['live', 'backtest'] = Field(

        description="Indicates whether the source is for live data or backtesting."
    )
    adjusted:bool

class Trade(BaseObjectOrm):
    @classmethod
    def create_or_update(cls, trade_kwargs,timeout=None) -> None:
        url = f"{cls.get_object_url()}/create_or_update/"
        data = cls.serialize_for_json(trade_kwargs)
        payload = {"json": data}
        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="POST", url=url, payload=payload,
                         timeout=timeout)
        if r.status_code in [200] == False:
            raise Exception(f" {r.text()}")
        return r



class OrdersExecutionConfiguration(BaseModel):
    broker_class: str
    broker_configuration: dict

class TargetPortfolioExecutionConfiguration(BaseObjectOrm,BaseVamPydanticModel):
    related_portfolio: Optional[int]=None
    portfolio_build_configuration: Optional[Dict[str, Any]] = None
    orders_execution_configuration: Optional[OrdersExecutionConfiguration] = None

    rebalance_tolerance_percent: float = Field(default=0.02, ge=0)
    minimum_notional_for_a_rebalance: float = Field(default=15.00, ge=0)
    max_latency_in_cdc_seconds: float = Field(default=60.00, ge=0)
    unwind_funds_hanging_limit_seconds: Optional[float] = None
    minimum_positions_holding_seconds: Optional[float] = None
    rebalance_step_every_seconds: Optional[float] = None
    max_data_latency_seconds: Optional[float] = None


class TargetPortfolio(BaseObjectOrm, BaseVamPydanticModel):
    id: Optional[Union[int,str]] = None
    portfolio_name: str = Field(..., max_length=255)
    portfolio_ticker: str = Field(..., max_length=150)
    latest_rebalance: Optional[datetime.datetime] = None
    calendar:Calendar
    
    
    is_asset_only: bool = False
    build_purpose:str
    is_active: bool = False
    local_time_serie_id: int = Field(...,)
    local_time_serie_hash_id: str = Field(...)
    local_signal_time_serie_id:int = Field(...,)

    builds_from_predictions: bool = False
    builds_from_target_positions: bool = False
    follow_account_rebalance: bool = False
    tracking_funds_expected_exposure_from_latest_holdings: bool = False
    available_in_venues: List[Union[int, ExecutionVenue]]
    latest_weights:Optional[HistoricalWeights] =None

    creation_date: Optional[datetime.datetime] = None
    execution_configuration: Union[int, TargetPortfolioExecutionConfiguration]



    @classmethod
    def create_from_time_series(cls,  portfolio_name: str,
                            build_purpose: str,
                            local_time_serie_id: int,
                            local_time_serie_hash_id: str,
                            local_signal_time_serie_id:int,
                            is_active: bool ,
                            available_in_venues__symbols:list[str],
                            execution_configuration: dict,
                            calendar_name: str,
                            tracking_funds_expected_exposure_from_latest_holdings:bool
                                ) -> "PortfolioType":
        url = f"{cls.get_object_url()}/create_from_time_series/"
        # Build the payload with the required arguments.
        payload_data = {
            "portfolio_name": portfolio_name,
            "build_purpose": build_purpose,
            "is_active": is_active,
            "local_time_serie_id": local_time_serie_id,
            "local_time_serie_hash_id": local_time_serie_hash_id,
            # Using the same ID for local_signal_time_serie_id as specified.
            "local_signal_time_serie_id": local_signal_time_serie_id,
            "available_in_venues__symbols": available_in_venues__symbols,
            "execution_configuration": execution_configuration,
            "calendar_name": calendar_name,
            "tracking_funds_expected_exposure_from_latest_holdings":tracking_funds_expected_exposure_from_latest_holdings
        }

        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="POST", url=url, payload={"json": payload_data},)
        if r.status_code not in [201]:
            raise Exception(f" {r.json()}")

        return cls(**r.json())

    def add_venue(self,venue_id)->None:
        url = f"{self.get_object_url()}/{self.id}/add_venue/"
        payload = {"json": {"venue_id": venue_id}}
        r = make_request(s=self.build_session(),loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload)
        if r.status_code in [200] == False:
            raise Exception(f" {r.text()}")


class PortfolioTags(BaseVamPydanticModel):
    id:Optional[int]=None
    name:str
    color:str
    
class TargetPortfolioFrontEndDetails(BaseObjectOrm, BaseVamPydanticModel):
    target_portfolio: Optional[TargetPortfolio]=None
    comparable_portfolios: Optional[List[int]] = None
    backtest_table_time_index_name: Optional[str] = Field(None, max_length=20)
    backtest_table_price_column_name: Optional[str] = Field(None, max_length=20)
    tags: Optional[List[PortfolioTags]] = None

    @classmethod
    def get_object_url(cls):
        url = f"{cls.ROOT_URL.replace('orm/api', 'api')}/{cls.END_POINTS[cls.class_name()]}"
        return url

    @property
    def id(self):
        return self.target_portfolio.id
    
    @staticmethod
    def get_base_endpoint():
        return  VAM_API_ENDPOINT.replace("orm/api","api/")

    @classmethod
    def create(cls, *args, **kwargs):
        """

        :return:
        :rtype:
        """
        base_url = cls.get_base_endpoint()+"target-portfolio-details"
        data = cls.serialize_for_json(kwargs)
        payload = {"json": data}

        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="POST", url=f"{base_url}/", payload=payload)
        if r.status_code not in [201, 200]:
            raise Exception(r.text)
        return cls(**r.json())

    @classmethod
    def create_or_update(cls, *args, **kwargs):
        """

        :return:
        :rtype:
        """
        base_url = cls.get_base_endpoint() + "target-portfolio-details/create_or_update"
        data = cls.serialize_for_json(kwargs)
        payload = {"json": data}

        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="POST", url=f"{base_url}/", payload=payload)
        if r.status_code not in [201, 200]:
            raise Exception(r.text)
        return cls(**r.json())

    @classmethod
    def get_or_build_asset_only_portfolio(cls, timeout=None,*args,**kwargs):
        url = cls.get_base_endpoint() + "target-portfolio-details/get_or_build_asset_only_portfolio"
        payload = {"json": kwargs }
        r = make_request(s=cls.build_session(),
                         loaders=cls.LOADERS, r_type="POST", url=f"{url}/", payload=payload, timeout=timeout)
        if r.status_code not in [200, 201, 204]:
            raise Exception(f"Error inserting new prediction{r.text}")
        return r.json()
class AssetOnlyPortfolio(BaseObjectOrm):
    pass

class ExecutionPrediction(BaseObjectOrm):
    @classmethod
    def add_prediction_from_time_serie(cls, time_serie_hash_id: str,
                                    prediction_time: datetime.datetime, symbol_to_search_map,
                                    predictions: dict, human_readable_name: Union[None, str] = None,timeout=None):
        """

        :param session:
        :return:
        """
        url = f"{cls.get_object_url()}/add_prediction_from_time_serie/"
        payload = {"json": {"time_serie_hash_id": time_serie_hash_id,

                            "prediction_time": prediction_time.strftime(DATE_FORMAT),
                            "symbol_to_search_map": symbol_to_search_map,
                            "predictions": predictions, "human_readable_name": human_readable_name,

                            }, }

        r = make_request(s=cls.build_session(),
                         loaders=cls.LOADERS, r_type="POST", url=url, payload=payload,timeout=timeout)
        if r.status_code in [201, 204] == False:
            raise Exception(f"Error inserting new prediction{r.text}")
        return r.json()


class VirtualFundPositionDetail(BaseObjectOrm, BaseVamPydanticModel):
    id: Optional[int] = None
    asset: Union[Asset,AssetFutureUSDM,int]
    price: float
    quantity: float
    parents_holdings: Union[int,"VirtualFundHistoricalHoldings"]

    @property
    def asset_id(self):
        return self.asset if isinstance(self.asset,int) else self.asset.id

    @root_validator(pre=True)
    def resolve_assets(cls, values):
        # Check if 'asset' is a dict and determine its type
        if isinstance(values.get('asset'), dict):
            asset = values.get('asset')
            asset = resolve_asset(asset_dict=asset)
            values['asset'] = asset

        return values

class VirtualFundHistoricalHoldings(BaseObjectOrm, BaseVamPydanticModel):
    related_fund: Union["VirtualFund",int]  # assuming VirtualFund is another Pydantic model
    target_trade_time: Optional[datetime.datetime] = None
    target_weights: Optional[dict] = Field(default=None)
    is_trade_snapshot: bool = Field(default=False)
    fund_account_target_exposure: float = Field(default=0)
    fund_account_units_exposure: Optional[float] = Field(default=None)
    holdings:list[VirtualFundPositionDetail]

class ExecutionQuantity(BaseModel):
    asset: Union[Asset,AssetFutureUSDM,  int]
    quantity: float
    reference_price:Union[None,float]

    def __repr__(self):
        return f"{self.__class__.__name__}(asset={self.asset}, quantity={self.quantity})"

    @root_validator(pre=True)
    def resolve_assets(cls, values):
        # Check if 'asset' is a dict and determine its type
        if isinstance(values.get('asset'), dict):
            asset = values.get('asset')
            asset = resolve_asset(asset_dict=asset)
            values['asset'] = asset

        return values


class TargetRebalance(BaseModel):
    target_execution_positions: ExecutionPositions
    execution_target: List[ExecutionQuantity]

    @property
    def rebalance_asset_map(self):
        return  {e.asset.id: e.asset for e in self.execution_target}


class VirtualFund(BaseObjectOrm, BaseVamPydanticModel):
    id: Optional[float] = None
    target_portfolio: Union[int,"TargetPortfolio"]
    target_account: AccountMixin
    fund_name: str
    fund_comments: str
    notional_exposure_in_account: float
    latest_holdings: "VirtualFundHistoricalHoldings" = None
    latest_rebalance: Optional[datetime.datetime] = None
    fund_nav: float = Field(default=0)
    fund_nav_date: Optional[datetime.datetime] = None
    requires_nav_adjustment: bool = Field(default=False)
    target_portfolio_weight_in_account: Optional[float] = None
    last_trade_time: Optional[datetime.datetime] = None
    latest_holdings_are_only_cash: bool

    def sanitize_target_weights_for_execution_venue(self,target_weights:dict):
        """
        This functions switches assets from main net to test net to guarante consistency in the recording
        of trades and orders
        Args:
            target_weights:{asset_id:WeightExecutionPosition}

        Returns:

        """
        if self.target_account.execution_venue.symbol == CONSTANTS.BINANCE_TESTNET_FUTURES_EV_SYMBOL:
            target_ev=CONSTANTS.BINANCE_TESTNET_FUTURES_EV_SYMBOL
            new_target_weights={}
            for _, position in target_weights.items():
                AssetClass = position.asset.__class__
                asset,_ = AssetClass.filter(symbol=position.asset.unique_symbol, execution_venue__symbol=target_ev,
                                        asset_type=position.asset.asset_type,
                                        )
                asset = asset[0]
                new_position = copy.deepcopy(position)
                new_position.asset=asset
                new_target_weights[asset.id] = new_position
                # todo create in DB an execution position
        else:
            new_target_weights = target_weights

        return new_target_weights




    def build_rebalance_from_target_weights(self, target_execution_postitions: ExecutionPositions,
                                            positions_prices: dict(), absolute_rebalance_weight_limit=.02
                                            ) -> TargetRebalance:
        """

        Returns
        -------

        """
        actual_positions = {}
        target_weights = {p.asset_id: p for p in target_execution_postitions.positions}
        #substitute target weights in case of testnets
        target_weights=self.sanitize_target_weights_for_execution_venue(target_weights)




        positions_to_rebalance = []
        if self.latest_holdings is not None:
            actual_positions = {p.asset_id : p for p in self.latest_holdings.holdings}

            # positions to unwind first
            positions_to_unwind=[]
            for position in self.latest_holdings.holdings:
                if position.quantity==0.0:
                    continue
                if position.asset_id not in target_weights.keys():
                    positions_to_unwind.append(ExecutionQuantity(asset=position.asset,
                                                     reference_price=None,
                                                     quantity=-position.quantity))

            positions_to_rebalance.extend(positions_to_unwind)

        for target_position in target_execution_postitions.positions:
            price = positions_prices[target_position.asset_id]

            current_weight, current_position = 0, 0
            if target_position.asset_id in actual_positions.keys():
                current_weight = actual_positions[target_position.asset_id].quantity * price / self.notional_exposure_in_account
                current_position = actual_positions[target_position.asset_id].quantity
            target_weight = target_position.weight_notional_exposure
            if abs(target_weight - current_weight) <= absolute_rebalance_weight_limit:
                continue
            target_quantity = self.notional_exposure_in_account * target_position.weight_notional_exposure / price
            rebalance_quantity = target_quantity - current_position
            positions_to_rebalance.append(ExecutionQuantity(asset=target_position.asset,
                                                            quantity=rebalance_quantity,
                                                            reference_price=price
                                                            ))

        target_rebalance = TargetRebalance(target_execution_positions=target_execution_postitions,
                                           execution_target=positions_to_rebalance
                                           )
        return target_rebalance

    @validator('last_trade_time', pre=True, always=True)
    def parse_last_trade_time(cls, value):
        value = validator_for_string(value)
        return value

    @validator('fund_nav_date', pre=True, always=True)
    def parse_fund_nav_date(cls, value):
        value = validator_for_string(value)
        return value

    @validator('latest_rebalance', pre=True, always=True)
    def parse_latest_rebalance(cls, value):
        value = validator_for_string(value)
        return value

    def get_account(self):
        a, r = Account.get(id=self.target_account)
        return a

    def get_latest_trade_snapshot_holdings(self):
        url = f"{self.get_object_url()}/{int(self.id)}/get_latest_trade_snapshot_holdings"
        r = make_request(s=self.build_session(),
                         loaders=self.LOADERS, r_type="GET", url=url)

        if r.status_code != 200:
            raise HtmlSaveException(r.text)
        if len(r.json()) == 0:
            return None
        return VirtualFundHistoricalHoldings(**r.json())



class OrderStatus(str, Enum):
    LIVE = "live"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    NOT_PLACED = "not_placed"
class OrderSide(IntEnum):
    SELL = -1
    BUY = 1

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    NOT_PLACED = "not_placed"

class Order(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int] = Field(None, primary_key=True)
    order_remote_id: str
    order_time: datetime.datetime
    order_side: OrderSide  # Use int for choices (-1: SELL, 1: BUY)
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # Price can be None for market orders
    status: OrderStatus = OrderStatus.NOT_PLACED
    filled_quantity: Optional[float] = 0.0
    filled_price: Optional[float] = None
    order_manager: Optional[int] = None  # Assuming foreign key ID is used
    asset: int  # Assuming foreign key ID is used
    related_fund: Optional[int] = None  # Assuming foreign key ID is used
    related_account: int  # Assuming foreign key ID is used
    comments: Optional[str] = None

    class Config:
        use_enum_values = True  # This allows using enum values directly

    @classmethod
    def create_or_update(cls,*args,**kwargs):
        base_url = cls.get_object_url()
        payload = {"json": kwargs }
        r = make_request(s=cls.build_session(),
                         loaders=cls.LOADERS, r_type="POST", url=f"{base_url}/create_or_update/", payload=payload)
        if r.status_code <200 or r.status_code>=300:
            raise Exception(r.text)
        return cls(**r.json())



class OrderManagerTargetQuantity(BaseObjectOrm,BaseVamPydanticModel):

    id:Optional[int]=None
    asset:Union[Asset,int]
    quantity:float

class OrderManager(BaseObjectOrm,BaseVamPydanticModel):
    id: Optional[int] = None
    target_time: datetime.datetime
    target_rebalance: list[OrderManagerTargetQuantity]
    order_received_time: Optional[datetime.datetime] = None
    execution_end: Optional[datetime.datetime] = None
    related_account: Union[Account,int]  # Representing the ForeignKey field with the related account ID

    @classmethod
    def destroy_before_date(cls,target_date:datetime.datetime):
        base_url = cls.get_object_url()
        payload = {"json": {"target_date": target_date.strftime(DATE_FORMAT),
                         },}


        r = make_request(s=cls.build_session(),
                         loaders=cls.LOADERS,r_type="POST", url=f"{base_url}/destroy_before_date/", payload=payload)

        if r.status_code != 204:
            raise Exception(r.text)


class ExecutionError(BaseObjectOrm):
    pass
class InterfaceError(BaseObjectOrm):
    pass
