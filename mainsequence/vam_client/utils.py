import requests
import os
from requests.structures import CaseInsensitiveDict
from pathlib import Path
import datetime
import time
import pytz
from typing import Union
import logging


from mainsequence.vam_client.local_vault import get_all_entries_in_vault_for_venue
from tqdm import tqdm
VAM_ENDPOINT=os.environ.get('VAM_ENDPOINT')
VAM_API_ENDPOINT=f"{VAM_ENDPOINT}/orm/api"
VAM_REST_TOKEN_URL=f"{VAM_ENDPOINT}/auth/rest-token-auth/"

assert VAM_API_ENDPOINT is not None, "VAM_ENDPOINT environment variable has not been set"
VAM_ADMIN_USER=os.environ.get('VAM_ADMIN_USER')
VAM_ADMIN_PASSWORD=os.environ.get('VAM_ADMIN_PASSWORD')


def get_vam_client_logger():
    # Check if the logger with the name 'virtualfundbuilder' already exists
    logger = logging.getLogger('mainsequence.vam_client')

    # # If the logger doesn't have any handlers, create it using the custom function
    # if not logger.hasHandlers():
    #     logger_file = os.environ.get('VFB_LOGS_PATH', os.path.join(os.path.expanduser("~"), "virtualfundbuilder/logs"))
    #     logger = create_logger_in_path(logger_name="mainsequence.vam_client", logger_file=logger_file, application_name="mainsequence.vam_client")

    return logger

logger = get_vam_client_logger()

def request_to_datetime(string_date:str):
    try:
        date = datetime.datetime.strptime(string_date, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=pytz.utc)
    except ValueError:
        date = datetime.datetime.strptime(string_date, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=pytz.utc)
    return date

class DoesNotExist(Exception):
    pass

class AuthLoaders:

    def __init__(self):

        self.gcp_token_decoded=None
    @property
    def auth_headers(self):

        if self.gcp_token_decoded is not None:
            if ((self.gcp_token_decoded["exp"] - datetime.timedelta(minutes=10)) - datetime.datetime.utcnow()).total_seconds() < 0.0:
                self.refresh_headers()
        if hasattr(self, "_auth_headers") == False:
            self.refresh_headers()
        return self._auth_headers

    def refresh_headers(self):
        logger.debug("Getting Auth Headers ASSETS_ORM")
        self._auth_headers, gcp_token_decoded = get_authorization_headers()
        if gcp_token_decoded is not None:
            self.gcp_token_decoded = gcp_token_decoded
            self.gcp_token_decoded["exp"] = datetime.datetime.utcfromtimestamp(gcp_token_decoded["exp"])




def get_authorization_headers(token_url=VAM_REST_TOKEN_URL,
                          username=VAM_ADMIN_USER,password=VAM_ADMIN_PASSWORD):

    headers=get_rest_token_header(token_url=token_url,username=username ,password=password)
    return headers

def get_gcp_headers():
    import google.auth.transport.requests
    import google.oauth2.id_token
    from google.auth import jwt

    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req,
                                                     VAM_ENDPOINT)
    headers = {"X-Serverless-Authorization": f"Bearer {id_token}"}
    return headers, jwt.decode(id_token, verify=False)

def make_request(s, r_type: str, url: str,   loaders:Union[AuthLoaders,None], payload: Union[dict, None] = None,
                timeout=None
                 ):
    from requests.models import Response

    TIMEOFF = .25
    TRIES = 15 // TIMEOFF
    timeout=30 if timeout is None else timeout
    payload = {} if payload is None else payload
    def get_req(session):
        if r_type == "GET":
            req = session.get
        elif r_type == "POST":
            req = session.post
        elif r_type == "PATCH":
            req = session.patch
        elif r_type =="DELETE":
            req=session.delete
        else:
            raise NotImplementedError
        return  req
    req=get_req(session=s)
    keep_request = True
    counter = 0
    while keep_request == True:

        try:
            r = req(url, timeout=timeout, **payload)
            if r.status_code  in [403, 401]:
                logger.warning(f"Error {r.status_code} Refreshing headers")
                loaders.refresh_headers()
                s.headers.update(loaders.auth_headers)
                req = get_req(session=s)
            else:
                keep_request = False
                break
        except requests.exceptions.ConnectionError as errc:
            logger.warning(f"Error connecting {url} ", errc)

        except Exception as e:
            logger.warning(f"Error connecting {url} ", e)

        counter = counter + 1
        if counter >= TRIES:
            keep_request = False
            r = Response()
            r.code = "expired"
            r.error_type = "expired"
            r.status_code = 500
            break

        logger.debug(f"SLEEPING {TIMEOFF} to trying request again {counter} {url}")
        time.sleep(TIMEOFF)
    return r


def build_session():
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=2, )
    s.mount('http://', HTTPAdapter(max_retries=retries))
    return s


def get_constants(root_url=VAM_API_ENDPOINT):
    url = f"{root_url}/constants"
    loaders = AuthLoaders()
    s = build_session()
    s.headers.update(loaders.auth_headers)
    r = make_request(s=s,loaders=loaders, r_type="GET", url=url)

    return r.json()

def get_binance_constants(root_url=VAM_API_ENDPOINT):
    url = f"{ root_url}/binance/constants"
    loaders = AuthLoaders()
    s = build_session()
    s.headers.update(loaders.auth_headers)
    r = make_request(s=s,loaders=loaders, r_type="GET", url=url)

    return r.json()

class LazyConstants(dict):
    """
    Class Method to load constants only once they are called. this minimizes the calls to the API
    """
    CONSTANTS_METHOD=lambda self: get_constants()

    def __getattr__(self, key):
        CONSTANTS = self.CONSTANTS_METHOD()
        for tmp_key, value in CONSTANTS.items():
            self.__dict__[tmp_key] = value
            setattr(self, tmp_key, value)

        return self.__dict__[key]
class BinanceLazyConstants(LazyConstants):
    CONSTANTS_METHOD = lambda self: get_binance_constants()


if 'CONSTANTS' not in locals():
    CONSTANTS = LazyConstants()
if "BINANCE_CONSTANTS" not in locals():
    BINANCE_CONSTANTS = BinanceLazyConstants()


def get_rest_token_header(token_url:str,  username:str,password:str  ):
    if token_url is None:
        return None

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"

    if os.getenv("MAINSEQUENCE_TOKEN"):
        headers["Authorization"] = "Token " + os.getenv("MAINSEQUENCE_TOKEN")
        return headers, None
    else:
        raise Exception("MAINSEQUENCE_TOKEN is not set in env")

    s = build_session()
    gcp_auth = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
    gcp_token_decoded = None
    if gcp_auth is not None:
        if gcp_auth.strip() !="":
            gcp_headers ,gcp_token_decoded=get_gcp_headers()
            headers.update(gcp_headers)
            s.headers.update(gcp_headers)

    payload = dict(json={"username": username,
                         "password": password})
    response = make_request(s=s, url=token_url, r_type="POST", loaders=None, payload=payload)
    if response.status_code != 200:
        raise Exception(response.text)

    headers["Content-Type"] = "application/json"
    headers["Authorization"] = "Token " + response.json()["token"]

    return headers,gcp_token_decoded


def build_account_for_venue(execution_venue, account_id, api_key, api_secret):
    """
    Creates the account for a venue
    """
    from mainsequence.vam_client.models_binance import build_binance_futures_account_from_key
    from mainsequence.vam_client.models_alpaca import build_alpaca_account_from_keys
    BINANCE_FUTURES_VENUES = [CONSTANTS.BINANCE_TESTNET_FUTURES_EV_SYMBOL, CONSTANTS.BINANCE_FUTURES_EV_SYMBOL]

    if execution_venue.symbol in BINANCE_FUTURES_VENUES:
        build_binance_futures_account_from_key(
            execution_venue=execution_venue,
            account_id=account_id,
            api_key=api_key,
            api_secret=api_secret
        )
    elif execution_venue.symbol in CONSTANTS.ALPACA_VENUES:
        build_alpaca_account_from_keys(
            execution_venue=execution_venue,
            account_id=account_id,
            api_key=api_key,
            api_secret=api_secret,
        )
    else:
        raise NotImplementedError(f"Execution venue {execution_venue} not supported")


def build_accounts_for_venues(venue_symbols, rebuild_assets=True):
    """
    Creates all the accounts for a list of venues in VAM
    """
    for venue_symbol in venue_symbols:
        ev = get_venues_in_vault_from_symbol(venue_symbol)
        accounts_for_venue = get_all_entries_in_vault_for_venue(venue_symbol)

        # build assets using credentials of first account
        if rebuild_assets and len(accounts_for_venue) > 0:
            secrets = accounts_for_venue[0]["secrets"]
            build_assets_for_venue(
                execution_venue=ev,
                api_key=secrets["api_key"],
                api_secret=secrets["secret_key"]
            )

        # create all the accounts for the venue
        for i, account_entry in tqdm(enumerate(accounts_for_venue), f"Build accounts for venue {venue_symbol}"):
            secrets = account_entry["secrets"]
            build_account_for_venue(
                execution_venue=ev,
                account_id=account_entry["account_id"],
                api_key=secrets["api_key"],
                api_secret=secrets["secret_key"],
            )

def get_venue_from_symbol(execution_venue_symbol):
    from mainsequence.vam_client.models import ExecutionVenue

    ev, _ = ExecutionVenue.filter(symbol=execution_venue_symbol)
    if len(ev) == 0:
        # create if not exists
        ev = ExecutionVenue.create(symbol=execution_venue_symbol,
                                   name=CONSTANTS.EXECUTION_VENUES_NAMES[execution_venue_symbol])
    else:
        ev = ev[0]
    return ev