import datetime

import requests
from requests.structures import CaseInsensitiveDict
import json
import os
import tempfile
import pandas as pd
import psycopg2
from psycopg2 import errors
from psycopg2.extras import execute_batch
from pgcopy import CopyManager
from typing import Union, List
import time
import traceback
from joblib import Parallel, delayed
import psutil
import socket
import logging
import subprocess
from mainsequence.tdag.logconf import create_logger_in_path

TDAG_ENDPOINT = f"{os.environ.get('TDAG_ENDPOINT')}"

TDAG_TOKEN_URL = f"{os.environ.get('TDAG_ENDPOINT')}/auth/rest-token-auth/"

COMPRESSED_KEYS = ["json_compressed_", "jcomp_"]

DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


def get_tdag_client_logger():
    # Check if the logger with the name 'virtualfundbuilder' already exists
    logger = logging.getLogger('mainsequence.vam_client')

    # If the logger doesn't have any handlers, create it using the custom function
    if not logger.hasHandlers():
        logger_file = os.environ.get('VFB_LOGS_PATH', os.path.join(os.path.expanduser("~"), "virtualfundbuilder/logs"))
        logger = create_logger_in_path(logger_name="mainsequence.vam_client", logger_file=logger_file, application_name="mainsequence.vam_client")

    return logger

logger = get_tdag_client_logger()

def get_network_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # Connect to a well-known external host (Google DNS) on port 80
        s.connect(("8.8.8.8", 80))
        # Get the local IP address used to make the connection
        network_ip = s.getsockname()[0]
    return network_ip


def is_process_running(pid: int) -> bool:
    """
    Check if a process with the given PID is running.

    Args:
        pid (int): The process ID to check.

    Returns:
        bool: True if the process is running, False otherwise.
    """
    try:
        # Check if the process with the given PID is running
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        # Process with the given PID does not exist
        return False

def inflate_json_compresed_column(compressed_series: pd.Series):
    """
    Inflates a compressed json_compressed series
    Parameters
    ----------
    compressed_series : 

    Returns
    -------

    """
    uncompressed_data = compressed_series.values
    if isinstance(uncompressed_data[0], str):
        uncompressed_data = list(map(json.loads, uncompressed_data))
    else:
        uncompressed_data = uncompressed_data.tolist()
    inflated_data = pd.DataFrame(index=compressed_series.index,
                                 data=uncompressed_data)
    return inflated_data


class AuthLoaders:

    def __init__(self, time_series_orm_token_url=TDAG_TOKEN_URL,
                 time_series_orm_admin_user=None,
                 time_series_orm_admin_password=None,
                 gcp_credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)):

        self.time_series_orm_token_url = time_series_orm_token_url
        self.time_series_orm_admin_user = time_series_orm_admin_user
        self.time_series_orm_admin_password = time_series_orm_admin_password
        self.gcp_credentials_path = gcp_credentials_path
        if gcp_credentials_path is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path
        self.gcp_token_decoded = None

    @property
    def auth_headers(self):

        if self.gcp_token_decoded is not None:
            if ((self.gcp_token_decoded["exp"] - datetime.timedelta(
                    minutes=10)) - datetime.datetime.utcnow()).total_seconds() < 0.0:
                self.refresh_headers()
        if hasattr(self, "_auth_headers") == False:
            self.refresh_headers()
        return self._auth_headers

    def refresh_headers(self):
        logger.debug("Getting Auth Headers TS_ORM")
        self._auth_headers, gcp_token_decoded = get_authorization_headers(
            time_series_orm_token_url=self.time_series_orm_token_url,
            time_series_orm_admin_user=self.time_series_orm_admin_user,
            time_series_orm_admin_password=self.time_series_orm_admin_password,
            gcp_credentials_path=self.gcp_credentials_path
            )
        logger.debug("Auth Headers Acquired")
        if gcp_token_decoded is not None:
            self.gcp_token_decoded = gcp_token_decoded
            self.gcp_token_decoded["exp"] = datetime.datetime.utcfromtimestamp(gcp_token_decoded["exp"])


def get_gcp_headers(time_series_orm_root_url):
    import google.auth.transport.requests
    import google.oauth2.id_token
    from google.auth import jwt
    from google.auth.exceptions import TransportError

    TIMEOFF = .25
    TRIES = 15 // TIMEOFF
    auth_req = google.auth.transport.requests.Request()

    keep_request = True
    counter = 0
    while keep_request == True:
        try:
            id_token = google.oauth2.id_token.fetch_id_token(auth_req, time_series_orm_root_url)
            keep_request = False
            break
        except TransportError as te:
            logger.warning("Transport error - retrying")
        except Exception as e:
            raise e
        counter = counter + 1
        if counter >= TRIES:
            keep_request = False
            raise Exception("Couldnt retrive the GCP token")
        logger.info(f"SLEEPING {TIMEOFF} trying to request again google headers {counter} {time_series_orm_root_url}")
        time.sleep(TIMEOFF)
    headers = {"X-Serverless-Authorization": f"Bearer {id_token}"}
    return headers, jwt.decode(id_token, verify=False)


def make_request(s, r_type: str, url: str, loaders: Union[AuthLoaders, None], payload: Union[dict, None] = None,
                 time_out: Union[int, None] = None,
                 ):
    from requests.models import Response

    TIMEOFF = .25
    TRIES = 15 // TIMEOFF
    payload = {} if payload is None else payload

    if r_type == "GET":
        req = s.get
    elif r_type == "POST":
        req = s.post
    elif r_type == "PATCH":
        req = s.patch
    else:
        raise NotImplementedError
    read_timeout = 45 if time_out is None else time_out
    connection_timeout = 45
    keep_request = True
    counter = 0
    while keep_request == True:

        try:
            request_start = time.time()
            r = req(url, timeout=(connection_timeout, read_timeout), **payload)
            if r.status_code in [403, 401] and loaders is not None:
                logger.warning(f"ERROR {r.status_code} Refreshing headers")
                loaders.refresh_headers()
                time.sleep(1)
            else:
                keep_request = False
                break
        except requests.exceptions.ConnectionError as errc:
            logger.warning(f"ERROR req time {time.time() - request_start} Connection {url}  {errc}")


        except Exception as e:
            logger.warning(f"ERROR req time {time.time() - request_start} Connection {url} {e}")

        counter = counter + 1
        if counter >= TRIES:
            keep_request = False
            r = Response()
            r.code = "expired"
            r.error_type = "expired"
            r.status_code = 500

            break
        logger.info(f"SLEEPING {TIMEOFF} to trying request again {counter} {url}")
        # traceback.print_stack()
        time.sleep(TIMEOFF)
    return r


def build_session():
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=2, )
    s.mount('http://', HTTPAdapter(max_retries=retries))
    return s


def get_authorization_headers(time_series_orm_token_url: str,
                              time_series_orm_admin_user: str,
                              time_series_orm_admin_password: str,
                              gcp_credentials_path: Union[str, None]):
    gcp_token_decoded = None
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    if os.getenv("MAINSEQUENCE_TOKEN"):
        headers["Authorization"] = "Token " + os.getenv("MAINSEQUENCE_TOKEN")
        return headers, None
    else:
        raise Exception("MAINSEQUENCE_TOKEN is not set in env")
    try:

        s = build_session()
        if gcp_credentials_path is not None:
            if gcp_credentials_path.strip() != "":
                logger.debug("Requesting GCP Auth")
                gcp_headers, gcp_token_decoded = get_gcp_headers(
                    time_series_orm_root_url=time_series_orm_token_url.replace("/auth/rest-token-auth/", ""))
                headers.update(gcp_headers)
                s.headers.update(gcp_headers)
                logger.debug("GCP Auth Acquired")

        payload = dict(json={"username": time_series_orm_admin_user,
                             "password": time_series_orm_admin_password})

        response = make_request(s=s, url=time_series_orm_token_url, r_type="POST", loaders=None, payload=payload)

    except requests.exceptions.ConnectionError as e:
        raise Exception(f"Connection Error is it {time_series_orm_token_url} running?")
    if response.status_code != 200:
        raise Exception(f"Request Error is it {time_series_orm_token_url} {response.text}")

    headers["Authorization"] = "Token " + response.json()["token"]
    return headers, gcp_token_decoded


def get_constants(root_url=TDAG_ENDPOINT):
    url = f"{root_url}/orm/api/constants"
    loaders = AuthLoaders()
    s = build_session()
    s.headers.update(loaders.auth_headers)
    r = make_request(s=s, loaders=loaders, r_type="GET", url=url)

    return r.json()

class LazyConstants(dict):
    """
    Class Method to load constants only once they are called. this minimizes the calls to the API
    """

    def __getattr__(self, key):
        CONSTANTS = get_constants()
        for tmp_key, value in CONSTANTS.items():
            self.__dict__[tmp_key] = value
            setattr(self, tmp_key, value)

        return self.__dict__[key]


if 'CONSTANTS' not in locals():
    CONSTANTS = LazyConstants()

def concatenate_tables_on_time_index(base_table_config: dict, table_map: dict,
                                     target_value: datetime.datetime,
                                     great_or_equal: bool):
    query = ""
    target_value = target_value.strftime(DATE_FORMAT)
    comp_sign = ">=" if great_or_equal is True else ">"
    max_tables = len(table_map.keys())
    time_index_name = base_table_config['time_index_name']
    all_cols = list(base_table_config['column_dtypes_map'].keys())
    col_order = ",".join(all_cols)
    for counter, (key, table_name) in enumerate(table_map.items()):

        query = query + f"""SELECT '{key}' AS "key",{col_order}  from {table_name} a_{counter} where "{time_index_name}" {comp_sign}'{target_value}' """
        if counter < max_tables - 1:
            query = query + " UNION ALL "

    results = execute_raw_query(query=query)
    all_cols = ["key"] + all_cols
    results = pd.DataFrame(columns=all_cols, data=results).set_index(time_index_name)
    return results


def execute_raw_query(query):
    with  psycopg2.connect(TDAG_ORM_DB_CONNECTION) as conn:
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
    return result


def read_one_value_from_table(hash_id):
    with  psycopg2.connect(TDAG_ORM_DB_CONNECTION) as conn:
        cur = conn.cursor()
        cur.execute(f"select * from {hash_id} limit 1")
        result = cur.fetchone()
    return result


def read_sql_tmpfile(query, time_series_orm_uri_db_connection: str):

    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
            query=query, head="HEADER"
        )
        # conn = db_engine.raw_connection()
        # cur = conn.cursor()
        with  psycopg2.connect(time_series_orm_uri_db_connection) as conn:
            # TEMP FOR FUCKED UP BELOW
            # cur = session.connection().connection.cursor()
            cur = conn.cursor()
            cur.copy_expert(copy_sql, tmpfile)
            tmpfile.seek(0)
            df = pd.read_csv(tmpfile, header=0)

        return df


def chunks(lst: list, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def individual_copy(records: list, time_series_orm_db_connection: str, table_name: str, columns: list):
    with  psycopg2.connect(time_series_orm_db_connection) as conn:
        cur = conn.cursor()
        try:
            mgr = CopyManager(conn, f"public.{table_name}", columns)
            mgr.copy(records)
        except Exception as e:
            cur.rollback()
            raise e

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from typing import Union
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import itertools
import csv
from io import StringIO
from tqdm import tqdm  # Import tqdm for progress bar


def recreate_indexes(table_name, table_index_names,time_series_orm_db_connection,logger):
    # Use a separate connection for index management
    with psycopg2.connect(time_series_orm_db_connection) as conn:
        # Set autocommit to True if using CREATE INDEX CONCURRENTLY
        conn.autocommit = True
        with conn.cursor() as cur:
            for index_name, index_info in table_index_names.items():
                index_type, index_q = index_info["type"], index_info["query"]
                logger.info(f"Creating index '{index_name}'...")
                start_time = time.time()
                q = f"CREATE {index_type}  {index_name} ON public.{table_name} {index_q};"
                cur.execute(q)
                end_time = time.time()
                time_taken = end_time - start_time
                logger.info(f"Index '{index_name}' created in {time_taken:.2f} seconds.")
        # Reset autocommit if needed
        conn.autocommit = False

def direct_table_update(table_name, serialized_data_frame: pd.DataFrame, overwrite: bool,
grouped_dates,
                        time_index_name: str, index_names: list, table_is_empty: bool,table_index_names:dict,
                        time_series_orm_db_connection: Union[str, None] = None,
                        use_chunks: bool = True, num_threads: int = 4):
    """
    Updates the database table with the given DataFrame.

    Parameters:
    - table_name: Name of the database table.
    - serialized_data_frame: DataFrame containing the data to insert.
    - overwrite: If True, existing data in the date range will be deleted before insertion.
    - time_index_name: Name of the time index column.
    - index_names: List of index column names.
    - table_is_empty: If True, the table is empty.
    - time_series_orm_db_connection: Database connection string.
    - use_chunks: If True, data will be inserted in chunks using threads.
    - num_threads: Number of threads to use when use_chunks is True.
    """

    columns = serialized_data_frame.columns.tolist()



    def drop_indexes(table_name, table_index_names):
        # Use a separate connection for index management
        with psycopg2.connect(time_series_orm_db_connection) as conn:
            with conn.cursor() as cur:
                for index_name in table_index_names.keys():
                    drop_index_query = f'DROP INDEX IF EXISTS "{index_name}";'
                    print(f"Dropping index '{index_name}'...")
                    cur.execute(drop_index_query)
            # Commit changes after all indexes are processed
            conn.commit()
            print("All specified indexes dropped successfully.")



        # Drop indexes before insertion

    duplicates_exist = serialized_data_frame.duplicated(subset=index_names).any()
    assert not duplicates_exist, f"Duplicates found in columns: {index_names}"

    #do not drop indices this is only done on inception
    # if serialized_data_frame.shape[0] > 1000000:
    #     duplicates_exist = serialized_data_frame.duplicated(subset=index_names).any()
    #     assert not duplicates_exist, f"Duplicates found in columns: {index_names}"
    #     drop_indexes(table_name, table_index_names)

    if overwrite and not table_is_empty:
        min_d = serialized_data_frame[time_index_name].min()
        max_d = serialized_data_frame[time_index_name].max()

        with psycopg2.connect(time_series_orm_db_connection) as conn:
            try:
                with conn.cursor() as cur:
                    GROUPED_DELETE_CONDITIONS = []

                    if len(index_names) > 1:


                        grouped_dates=grouped_dates.reset_index(level="execution_venue_symbol").rename(columns={"min":"start_time","max":"end_time"})
                        grouped_dates=grouped_dates.reset_index()
                        grouped_dates=grouped_dates.to_dict("records")

                        # Build the DELETE query
                        delete_conditions = []
                        for item in grouped_dates:
                            asset_symbol = item['asset_symbol']
                            execution_venue_symbol = item['execution_venue_symbol']
                            start_time = item['start_time']
                            end_time = item['end_time']

                            # Format timestamps as strings
                            start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S%z')
                            end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S%z')

                            # Escape single quotes
                            asset_symbol = asset_symbol.replace("'", "''")
                            execution_venue_symbol = execution_venue_symbol.replace("'", "''")

                            # Build the condition string
                            condition = f"({time_index_name} >= '{start_time_str}' AND {time_index_name} <= '{end_time_str}' " \
                                        f"AND asset_symbol = '{asset_symbol}' AND execution_venue_symbol = '{execution_venue_symbol}')"
                            delete_conditions.append(condition)
                            
                        # Combine all conditions using OR
                        where_clause = ' OR '.join(delete_conditions)
                        delete_query = f"DELETE FROM public.{table_name} WHERE {where_clause};"

                        # Execute the DELETE query
                        cur.execute(delete_query)
                    else:
                        # Build a basic DELETE query using parameterized values
                        delete_query = f"DELETE FROM public.{table_name} WHERE {time_index_name} >= %s AND {time_index_name} <= %s;"
                        cur.execute(delete_query, (min_d, max_d))

                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"An error occurred during deletion: {e}")
                raise

    if use_chunks:
        total_rows = len(serialized_data_frame)
        num_threads = min(num_threads, total_rows)
        chunk_size = int(np.ceil(total_rows / num_threads))

        # Generator to yield chunks without copying data
        def get_dataframe_chunks(df, chunk_size):
            for start_row in range(0, df.shape[0], chunk_size):
                yield df.iloc[start_row:start_row + chunk_size]

        # Progress bar for chunks
        total_chunks = int(np.ceil(total_rows / chunk_size))

        def insert_chunk(chunk_df):
            try:
                with psycopg2.connect(time_series_orm_db_connection) as conn:
                    with conn.cursor() as cur:
                        buffer_size = 10000  # Adjust based on memory and performance requirements
                        data_generator = chunk_df.itertuples(index=False, name=None)

                        total_records = len(chunk_df)
                        with tqdm(total=total_records, desc="Inserting records", leave=False) as pbar:
                            while True:
                                batch = list(itertools.islice(data_generator, buffer_size))
                                if not batch:
                                    break

                                # Convert batch to CSV formatted string
                                output = StringIO()
                                writer = csv.writer(output)
                                writer.writerows(batch)
                                output.seek(0)

                                copy_query = f"COPY public.{table_name} ({', '.join(columns)}) FROM STDIN WITH CSV"
                                cur.copy_expert(copy_query, output)

                                # Update progress bar
                                pbar.update(len(batch))

                    conn.commit()
            except Exception as e:
                print(f"An error occurred during insertion: {e}")
                raise

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(insert_chunk, get_dataframe_chunks(serialized_data_frame, chunk_size)),
                      total=total_chunks, desc="Processing chunks"))

    else:
        # Single insert using the same optimized method
        try:
            with psycopg2.connect(time_series_orm_db_connection) as conn:
                with conn.cursor() as cur:
                    buffer_size = 10000
                    data_generator = serialized_data_frame.itertuples(index=False, name=None)
                    total_records = len(serialized_data_frame)
                    with tqdm(total=total_records, desc="Inserting records") as pbar:
                        while True:
                            batch = list(itertools.islice(data_generator, buffer_size))
                            if not batch:
                                break
                            #
                            output = StringIO()
                            writer = csv.writer(output)
                            writer.writerows(batch)
                            output.seek(0)

                            copy_query = f"COPY public.{table_name} ({', '.join(columns)}) FROM STDIN WITH CSV"
                            cur.copy_expert(copy_query, output)




                            # Update progress bar
                            pbar.update(len(batch))

                conn.commit()
        except Exception as e:
            print(f"An error occurred during single insert: {e}")
            raise
    # do not rebuild  indices this is only done on inception
    # if serialized_data_frame.shape[0] > 500000:
    #     subprocess.Popen(
    #         [
    #             "python", "-m", "mainsequence.tdag", "create_indices_in_table",
    #             f"--table_name={table_name}",
    #             f'--table_index_names={json.dumps(table_index_names)}',
    #             f"--time_series_orm_db_connection={time_series_orm_db_connection}"
    #         ],
    #         stdout=subprocess.DEVNULL,
    #         stderr=subprocess.DEVNULL,
    #         close_fds=True,
    #         env=os.environ.copy(),  # Pass the environment variables
    #     )



def direct_table_update_OLD(table_name, serialized_data_frame: pd.DataFrame, overwrite: bool,
                        time_index_name: str, index_names: list, table_is_empty: bool,
                        time_series_orm_db_connection: Union[str, None] = None,

                        ):
    records = serialized_data_frame.values.tolist()
    columns = serialized_data_frame.columns.to_list()

    if overwrite == True and table_is_empty == False:

        min_d = serialized_data_frame[time_index_name].min()
        max_d = serialized_data_frame[time_index_name].max()

        GROUPED_DELETE_CONDITIONS = []

        if len(index_names) > 1:
            grouped_dates = serialized_data_frame.groupby(["asset_symbol", "execution_venue_symbol"])[
                time_index_name].agg(['min', 'max']).reset_index("execution_venue_symbol")
            grouped_dates["combo"] = grouped_dates["min"].astype(str) + "-" + grouped_dates["max"].astype(str) + "_" + \
                                     grouped_dates["execution_venue_symbol"]
            grouped_dates = grouped_dates.reset_index()
            for _, df in grouped_dates.groupby("combo"):
                tmp_symbols = [f"'{c}'" for c in df["asset_symbol"].unique().tolist()]
                tmp_symbols = ",".join(tmp_symbols)
                tmp_min, tmp_max, ev_symbol = df.iloc[0]["min"], df.iloc[0]["max"], df.iloc[0]["execution_venue_symbol"]

                DEL_Q = {
                    'start_time': tmp_min,
                    'end_time': tmp_max,
                    'asset_symbol': tmp_symbols,
                    'execution_venue_symbol': ev_symbol
                },

                GROUPED_DELETE_CONDITIONS.append(DEL_Q)

        else:
            GROUPED_DELETE_QUERIES = [
                f"DELETE from public.{table_name} where {time_index_name} >= '{min_d}' and {time_index_name} <='{max_d}'"]
        multi_insert = False
        with  psycopg2.connect(time_series_orm_db_connection) as conn:
            try:
                with conn.cursor() as cur:
                    execute_batch(cur, ";".join(GROUPED_DELETE_QUERIES), [])

                    if multi_insert == False:
                        mgr = CopyManager(conn, f"public.{table_name}", columns)
                        mgr.copy(records)
                # Commit the transaction if everything succeeds
                conn.commit()
            except Exception as e:
                # Rollback the transaction on any error
                conn.rollback()
                print(f"An error occurred: {e}")
                raise

        if multi_insert == True:
            njobs = os.getenv("TDAG_TABLE_JOBS_INSERT", 1)
            Parallel(n_jobs=njobs)(
                delayed(individual_copy)(records=r, time_series_orm_db_connection=time_series_orm_db_connection,
                                         table_name=table_name, columns=columns, ) for r in
                chunks(lst=records, n=njobs))



    else:
        with  psycopg2.connect(time_series_orm_db_connection) as conn:
            # write to database
            mgr = CopyManager(conn, f"public.{table_name}", columns)
            mgr.copy(records)
            # conn.commit()


def concatenate_ts(
    tables_to_concatenate: List[str],
    start_value: Union[datetime.datetime, None],
    end_value: Union[datetime.datetime, None],
    great_or_equal: bool,
    less_or_equal: bool,
    index_to_concat: List[str],
    time_series_orm_db_connection: Union[str, None] = None,
):


    with psycopg2.connect(time_series_orm_db_connection) as conn:
        params = []
        subqueries = []
        for i, table in enumerate(tables_to_concatenate):
            alias = f"t{i+1}"
            where_clauses = []
            if start_value is not None:
                lower_op = '>=' if great_or_equal else '>'
                where_clauses.append(f"time_index {lower_op} %s")
                params.append(start_value)
            if end_value is not None:
                upper_op = '<=' if less_or_equal else '<'
                where_clauses.append(f"time_index {upper_op} %s")
                params.append(end_value)
            where_clause = ''
            if where_clauses:
                where_clause = 'WHERE ' + ' AND '.join(where_clauses)
            subquery = f"""
            SELECT * FROM {table}
            {where_clause}
            """
            subqueries.append((alias, subquery.strip()))

        # Build the FROM clause with full outer joins using USING clause
        join_columns = ', '.join(index_to_concat)
        # Start with the first subquery and its alias
        from_alias, from_subquery = subqueries[0]
        from_clause = f"({from_subquery}) AS {from_alias}"
        for alias, subquery in subqueries[1:]:
            from_clause = f"({from_clause} FULL OUTER JOIN ({subquery}) AS {alias} USING ({join_columns}))"

        final_query = f"SELECT * FROM {from_clause}"
        df = pd.read_sql(final_query, conn, params=params)
        df = df.set_index(index_to_concat)
        return df


def get_mean_bid_ask_spread(interval: str, book_table_name: str, symbol_list: list,
                            bid_ask_rank=1,
                            timeout=60 * 2):
    symbol_query = ",".join([f"'{c}'" for c in symbol_list])
    QUERY = f"""
                
                    select symbol,sum(spread_quantity) total_spread_quantity,
                    avg(spread_quantity) mean_spread_quantity,avg(bid_ask_spread) average_bid_offer,
                    last(mid_price,time_index) as last_mid_price
                    
                    from (
                    
                    WITH RawBids As (
                    
                    SELECT time_index, symbol,
                         (json_array_elements(best_bids)->>0)::float as bid_price,
                            (json_array_elements(best_bids)->>1)::float as bid_size
                         FROM
                            {book_table_name}
                        where time_index > NOW() - INTERVAL '{interval}' AND  symbol in ({symbol_query})
                        
                    ),
                     RawAsks As (
                    
                    SELECT time_index, symbol,
                        (json_array_elements(best_asks)->>0)::float as ask_price,
                    (json_array_elements(best_asks)->>1)::float as ask_size
                         FROM
                           {book_table_name}
                        
                        where time_index > NOW() - INTERVAL '{interval}' AND  symbol in ({symbol_query})
                    ),
                    
                     RankedBids AS (
                        SELECT
                            time_index,
                            symbol,
                             bid_price,
                            bid_size,
                            ROW_NUMBER() OVER (PARTITION BY time_index, symbol ORDER BY bid_price DESC) AS bid_rank
                        FROM
                            RawBids
                        
                        
                        
                    ),
                    RankedAsks AS (
                        SELECT
                            time_index,
                            symbol,
                           ask_price,
                            ask_size,
                            ROW_NUMBER() OVER (PARTITION BY time_index, symbol ORDER BY ask_price ASC) AS ask_rank
                        FROM
                             RawAsks
                        
                    )
                    SELECT
                        b.time_index,
                        b.symbol,
                        b.bid_price,
                        b.bid_size,
                        a.ask_price,
                        a.ask_size,
                        (a.ask_price - b.bid_price)/b.bid_price AS bid_ask_spread,
                        LEAST( a.ask_size,  b.bid_size) as spread_quantity,
                        (a.ask_price + b.bid_price)/2 as mid_price
                    FROM
                        RankedBids b
                    JOIN
                        RankedAsks a ON b.time_index = a.time_index
                                     AND b.symbol = a.symbol
                                     AND b.bid_rank = {bid_ask_rank}
                                     AND a.ask_rank = {bid_ask_rank}
                    ORDER BY
                        b.time_index, b.symbol
                    
                    ) best_bid_ask group by (symbol)

    
    """
    with psycopg2.connect(TDAG_ORM_DB_CONNECTION, options=f'-c statement_timeout={timeout}') as conn:
        with conn.cursor() as cursor:
            cursor.execute(QUERY)
            result = cursor.fetchall()  #
            column_names = [desc[0] for desc in cursor.description]
    average_bid_offer = pd.DataFrame(data=result, columns=column_names)
    return average_bid_offer
