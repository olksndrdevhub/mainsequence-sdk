import pandas as pd
import psycopg2
import tempfile
import tqdm

from concurrent.futures import ThreadPoolExecutor
import itertools
import csv
from io import StringIO
from tqdm import tqdm  # Import tqdm for progress bar

import numpy as np
import json

from typing import Dict, List, Union
import datetime

from ..utils import DATE_FORMAT, make_request, set_types_in_table
import os

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


def filter_by_assets_ranges(table_name, asset_ranges_map, index_names, data_source, column_types):
    """
    Query time series data dynamically based on asset ranges.

    Args:
        table_name (str): The name of the table to query.
        asset_ranges_map (dict): A dictionary where keys are asset symbols and values are dictionaries containing:
                                 - 'start_date' (datetime): The start date of the range.
                                 - 'start_date_operand' (str): The SQL operand for the start date (e.g., '>=' or '>').
                                 - 'end_date' (datetime or None): The end date of the range.
        index_names (list): List of column names to set as the DataFrame index.
        data_source: A data source object with a method `get_connection_uri()` to get the database connection URI.

    Returns:
        pd.DataFrame: A Pandas DataFrame with the queried data, indexed by the specified columns.
    """
    # Base SQL query
    query_base = f"SELECT * FROM {table_name} WHERE"

    # Initialize a list to store query parts
    query_parts = []

    # Build query dynamically based on the asset_ranges_map dictionary
    for symbol, range_dict in asset_ranges_map.items():
        if range_dict['end_date'] is not None:
            tmp_query = (
                f" (asset_symbol = '{symbol}' AND "
                f"time_index BETWEEN '{range_dict['start_date']}' AND '{range_dict['end_date']}') "
            )
        else:
            tmp_query = (
                f" (asset_symbol = '{symbol}' AND "
                f"time_index {range_dict['start_date_operand']} '{range_dict['start_date']}') "
            )
        query_parts.append(tmp_query)

    # Combine all query parts using OR
    full_query = query_base + " OR ".join(query_parts)

    # Execute the query and load results into a Pandas DataFrame
    df = read_sql_tmpfile(full_query, time_series_orm_uri_db_connection=data_source.get_connection_uri())

    # set correct types for values
    df = set_types_in_table(df, column_types)

    # Set the specified columns as the DataFrame index
    df = df.set_index(index_names)

    return df


def direct_data_from_db(metadata: dict, connection_uri: str,
                        start_date: Union[datetime.datetime, None] = None,
                        great_or_equal: bool = True, less_or_equal: bool = True,
                        end_date: Union[datetime.datetime, None] = None,
                        columns: Union[list, None] = None,
                        asset_symbols: Union[list, None] = None
                        ):
    """
    Connects directly to the DB without passing through the ORM to speed up calculations.

    Parameters
    ----------
    metadata : dict
        Metadata containing table and column details.
    connection_config : dict
        Connection configuration for the database.
    start_date : datetime.datetime, optional
        The start date for filtering. If None, no lower bound is applied.
    great_or_equal : bool, optional
        Whether the start_date filter is inclusive (>=). Defaults to True.
    less_or_equal : bool, optional
        Whether the end_date filter is inclusive (<=). Defaults to True.
    end_date : datetime.datetime, optional
        The end date for filtering. If None, no upper bound is applied.
    columns : list, optional
        Specific columns to select. If None, all columns are selected.

    Returns
    -------
    pd.DataFrame
        Data from the table as a pandas DataFrame, optionally filtered by date range.
    """

    def fast_table_dump(connection_config, table_name, ):
        query = f"COPY {table_name} TO STDOUT WITH CSV HEADER"

        with psycopg2.connect(connection_config['connection_details']) as connection:
            with connection.cursor() as cursor:
                import io
                buffer = io.StringIO()
                cursor.copy_expert(query, buffer)
                buffer.seek(0)
                df = pd.read_csv(buffer)
                return df

    # Build the SELECT clause
    select_clause = ", ".join(columns) if columns else "*"

    # Build the WHERE clause dynamically
    where_clauses = []
    time_index_name = metadata['sourcetableconfiguration']['time_index_name']
    if start_date:
        operator = ">=" if great_or_equal else ">"
        where_clauses.append(f"{time_index_name} {operator} '{start_date}'")
    if end_date:
        operator = "<=" if less_or_equal else "<"
        where_clauses.append(f"{time_index_name} {operator} '{end_date}'")

    if asset_symbols:
        helper_symbol = "','"
        where_clauses.append(f"asset_symbol IN ('{helper_symbol.join(asset_symbols)}')")

    # Combine WHERE clauses
    where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Construct the query
    query = f"SELECT {select_clause} FROM {metadata['table_name']} {where_clause}"
    # if where_clause=="":
    #     data=fast_table_dump(connection_config, metadata['table_name'])
    #     data[metadata["sourcetableconfiguration"]['time_index_name']]=pd.to_datetime(data[metadata["sourcetableconfiguration"]['time_index_name']])
    # else:
    with psycopg2.connect(connection_uri) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            column_names = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()

    # Convert to DataFrame
    data = pd.DataFrame(data=data, columns=column_names)

    data = data.set_index(metadata['sourcetableconfiguration']["index_names"])

    return data


def direct_table_update(table_name, serialized_data_frame: pd.DataFrame, overwrite: bool,
                        grouped_dates,
                        time_index_name: str, index_names: list, table_is_empty: bool, table_index_names: dict,
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

    # do not drop indices this is only done on inception
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

                        grouped_dates = grouped_dates.reset_index(level="execution_venue_symbol").rename(
                            columns={"min": "start_time", "max": "end_time"})
                        grouped_dates = grouped_dates.reset_index()
                        grouped_dates = grouped_dates.to_dict("records")

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


def process_and_update_table(
        serialized_data_frame,
        metadata: Dict,
        grouped_dates: List,
        data_source: object,
        index_names: List[str],
        time_index_name: str,
        logger: object,
        overwrite: bool = False,
        JSON_COMPRESSED_PREFIX: List[str] = None,


):
    """
    Process a serialized DataFrame, handle overwriting, and update a database table.

    Args:
        serialized_data_frame (pd.DataFrame): The DataFrame to process and update.
        metadata (dict): Metadata about the table, including table configuration.
        grouped_dates (list): List of grouped dates to assist with the update.
        data_source (object): A data source object with a `get_connection_uri` method.
        index_names (list): List of index column names.
        time_index_name (str): The name of the time index column.
        overwrite (bool): Whether to overwrite the table or not.
        JSON_COMPRESSED_PREFIX (list): List of prefixes to identify JSON-compressed columns.

    Returns:
        None
    """
    if "asset_symbol" in serialized_data_frame.columns:
        serialized_data_frame['asset_symbol'] = serialized_data_frame['asset_symbol'].astype(str)

    TDAG_ENDPOINT = f"{os.environ.get('TDAG_ENDPOINT')}"
    base_url = TDAG_ENDPOINT + "/orm/api/dynamic_table" #metadata.get("root_url")
    serialized_data_frame = serialized_data_frame.replace({np.nan: None})

    # Validate JSON-compressed columns
    for c in serialized_data_frame.columns:
        if any([t in c for t in JSON_COMPRESSED_PREFIX]):
            assert isinstance(serialized_data_frame[c].iloc[0], dict)

    # Encode JSON-compressed columns
    for c in serialized_data_frame.columns:
        if any([t in c for t in JSON_COMPRESSED_PREFIX]):
            serialized_data_frame[c] = serialized_data_frame[c].apply(lambda x: json.dumps(x).encode())

    # Handle overwrite and decompress chunks if required
    recompress = False
    if overwrite:
        url = f"{base_url}/{metadata['id']}/decompress_chunks/"
        from ..models import BaseObject
        s = BaseObject.build_session()

        r = make_request(
            s=s, loaders=BaseObject.LOADERS,
            r_type="POST",
            url=url,
            payload={
                "json": {
                    "start_date": serialized_data_frame[time_index_name].min().strftime(DATE_FORMAT),
                    "end_date": serialized_data_frame[time_index_name].max().strftime(DATE_FORMAT),
                }
            },
            time_out=60 * 5,
        )

        if r.status_code not in [200, 204]:
            logger.error(r.text)
            raise Exception("Error trying to decompress table")
        elif r.status_code == 200:
            recompress = True

    # Check if the table is empty
    table_is_empty = metadata["sourcetableconfiguration"]["last_time_index_value"] is None

    # Update the table
    direct_table_update(
        serialized_data_frame=serialized_data_frame,
        grouped_dates=grouped_dates,
        time_series_orm_db_connection=data_source.get_connection_uri(),
        table_name=metadata["hash_id"],
        overwrite=overwrite,
        index_names=index_names,
        time_index_name=time_index_name,
        table_is_empty=table_is_empty,
        table_index_names=metadata["table_index_names"],
    )

    # Recompress if needed
    if recompress:
        # Logic to recompress if needed (currently a placeholder)
        pass
