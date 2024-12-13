import pandas as pd
import datetime
from typing import Union, List,Dict

import os



from mainsequence.tdag.logconf import console_logger, create_logger_in_path

from mainsequence.tdag_client import (DynamicTableHelpers, TimeSerieNode, TimeSerieLocalUpdate,
                                      LocalTimeSeriesDoesNotExist,PodLocalLake,
                                      DynamicTableDoesNotExist, DynamicTableDataSource, CONSTANTS)

from mainsequence.tdag.logconf import get_tdag_logger

logger = get_tdag_logger()











class PersistManager:
    def __init__(self,data_source, local_hash_id: str, remote_table_hashed_name: Union[str, None],

                 logger:Union[str,None]=None, description: Union[str,None]=None,
                 class_name: Union[str, None] = None,
                 human_readable: Union[str, None] = None, metadata: Union[dict, None] = None,
                 local_metadata: Union[dict, None] = None

                 ):
        self.data_source=data_source
        self.local_hash_id = local_hash_id
        self.description = description or ""
        if local_metadata is not None and metadata is None:
            # query remote hash_id
            metadata = local_metadata["remote_table"]

        self.remote_table_hashed_name = remote_table_hashed_name
        self.logger = logger if logger is not None else console_logger("timescale_persist_manager",
                                                                       application_name="tdag")
        self.dth = DynamicTableHelpers(logger=logger)

        self.table_model_loaded = False
        self.human_readable = human_readable if human_readable is not None else local_hash_id

        self.class_name = class_name
        if self.local_hash_id is not None:
            self.synchronize_metadata(meta_data=metadata,
                                      local_metadata=local_metadata,
                                      class_name=class_name)

    @classmethod
    def get_from_data_type(self,data_source:DynamicTableDataSource, *args, **kwargs):



        data_type=data_source.data_type
        if data_type==CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE:
            return DataLakePersistManager(data_source=data_source,*args, **kwargs)

        elif data_type==CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB:
            return TimeScaleLocalPersistManager(data_source=data_source,*args, **kwargs)

    def depends_on_connect(self,new_ts:"TimeSerie"):
        """
        Connects a time Serie as relationship in the DB
        Parameters
        ----------
        new_ts :

        Returns
        -------

        """
        try:
            human_readable=new_ts.local_persist_manager.metadata['human_readable']
        except KeyError:
            human_readable=new_ts.human_readable
        self.dth.depends_on_connect(source_hash_id=self.metadata["hash_id"],
                                    target_hash_id=new_ts.remote_table_hashed_name,

                                    source_local_hash_id=self.local_metadata["local_hash_id"],
                                    target_local_hash_id=new_ts.local_hash_id,

                                    target_class_name=new_ts.__class__.__name__,
                                    target_human_readable=human_readable,

                                    source_data_source_id=self.data_source.id,
                                    target_data_source_id=new_ts.data_source.id

                                    )

    def display_mermaid_dependency_diagram(self):
        from IPython.core.display import display, HTML, Javascript

        response = TimeSerieLocalUpdate.get_mermaid_dependency_diagram(local_hash_id=self.local_hash_id,
                                                                       data_source_id=self.data_source.id
                                                                       )
        from IPython.core.display import display, HTML, Javascript
        mermaid_chart = response.get("mermaid_chart")
        metadata = response.get("metadata")
        # Render Mermaid.js diagram with metadata display
        html_template = f"""
           <div class="mermaid">
           {mermaid_chart}
           </div>
           <div id="metadata-display" style="margin-top: 20px; font-size: 16px; color: #333;"></div>
           <script>
               // Initialize Mermaid.js
               if (typeof mermaid !== 'undefined') {{
                   mermaid.initialize({{ startOnLoad: true }});
               }}

               // Metadata dictionary
               const metadata = {metadata};

               // Attach click listeners to nodes
               document.addEventListener('click', function(event) {{
                   const target = event.target.closest('div[data-graph-id]');
                   if (target) {{
                       const nodeId = target.dataset.graphId;
                       const metadataDisplay = document.getElementById('metadata-display');
                       if (metadata[nodeId]) {{
                           metadataDisplay.innerHTML = "<strong>Node Metadata:</strong> " + metadata[nodeId];
                       }} else {{
                           metadataDisplay.innerHTML = "<strong>No metadata available for this node.</strong>";
                       }}
                   }}
               }});
           </script>
           """

        return mermaid_chart

    def get_all_local_dependencies(self):
        depth_df = TimeSerieLocalUpdate.get_all_dependencies(hash_id=self.local_hash_id,
                                                             data_source_id=self.data_source.id
                                                             )
        return depth_df

    def get_all_dependencies_update_priority(self):
        depth_df = TimeSerieLocalUpdate.get_all_dependencies_update_priority(hash_id=self.local_hash_id,
                                                                             data_source_id=self.data_source.id
                                                                             )
        return depth_df

    def set_ogm_dependencies_linked(self):

        TimeSerieLocalUpdate.set_ogm_dependencies_linked(hash_id=self.local_hash_id,
                                                         data_source_id=self.data_source.id
                                                         )

    @property
    def update_details(self):
        if "localtimeserieupdatedetails" in self.local_metadata.keys():
            return self.local_metadata['localtimeserieupdatedetails']
        return None
    @property
    def source_table_configuration(self):
        if "sourcetableconfiguration" in self.metadata.keys():
            return self.metadata['sourcetableconfiguration']
        return None
    def update_source_informmation(self, git_hash_id:str, source_code:str):
        """

        Args:
            git_hash_id:
            source_code:

        Returns:

        """

        self.metadata = self.dth.patch(metadata=self.metadata, time_serie_source_code_git_hash=git_hash_id,
                            time_serie_source_code=source_code,)

    def set_last_index_value(self):
        return TimeSerieLocalUpdate.set_last_update_index_time(metadata=self.local_metadata)

    @staticmethod
    def batch_data_persisted(hash_id_list: list):

        exist = {}
        dth = DynamicTableHelpers()
        in_db, _ = dth.exist(hash_id__in=hash_id_list)

        for t in hash_id_list:

            if t in in_db:
                exist[t] = True
            else:
                exist[t] = False

        return exist

    def synchronize_metadata(self, meta_data: Union[dict, None], local_metadata: Union[dict, None],
                             set_last_index_value: bool = False,
                             class_name: Union[str, None] = None
                             ):
        """
        forces a synchronization between table and metadata
        :return:
        """
        # start with remote metadata
        if set_last_index_value == True:
            TimeSerieLocalUpdate.set_last_update_index_time(metadata=self.local_metadata)
        if meta_data is None or local_metadata is None:  # avoid calling 2 times the DB
            meta_data = {}
            try:
                local_metadata = {}  # set to empty in case not exist
                local_metadata = TimeSerieLocalUpdate.get(local_hash_id=self.local_hash_id,
                                                          data_source_id=self.data_source.id
                                                          )
                if len(local_metadata) == 0:
                    raise LocalTimeSeriesDoesNotExist
            except LocalTimeSeriesDoesNotExist:
                # could be localmetadata is none but table could exist
                try:
                    meta_data = self.dth.get(hash_id=self.remote_table_hashed_name,
                                             data_source__id=self.data_source.id
                                             )
                except DynamicTableDoesNotExist:
                    pass

        if len(local_metadata) != 0:
            self.local_build_configuration = local_metadata["build_configuration"]
            self.local_build_metadata = local_metadata["build_meta_data"]
            self.local_metadata = local_metadata

            # metadata should always exist
            meta_data = local_metadata["remote_table"]

        if len(meta_data) != 0:
            remote_build_configuration, remote_build_metadata = meta_data["build_configuration"], meta_data[
                "build_meta_data"]
            self.remote_build_configuration = remote_build_configuration
            self.remote_build_metadata = remote_build_metadata
            self.metadata = meta_data

    def add_tags(self, tags: list):
        if any([t not in self.local_metadata["tags"] for t in tags]) == True:
            TimeSerieLocalUpdate.add_tags(tags=tags, local_metadata=self.local_metadata)

    def destroy(self, delete_only_table: bool):
        self.dth.destroy(metadata=self.metadata, delete_only_table=delete_only_table)

    @property
    def persist_size(self):
        try:
            return self.metadata['table_size']
        except KeyError:
            return 0

    def time_serie_exist(self):
        return self.dth.time_serie_exist_in_db(self.remote_table_hashed_name)

    def metadata_registered_in_db(self):
        return self.dth.get(hash_id=self.remote_table_hashed_name)


    def get_data_source_connection_details(self,override_id:Union[int,None]=None):
        from mainsequence.tdag_client import DynamicTableDataSource
        override_id=override_id or self.metadata['data_source']["id"]

        return DynamicTableDataSource.get_data_source_connection_details(connection_id=override_id)
    def patch_build_configuration(self,local_configuration:dict,remote_configuration:dict,
                                  remote_build_metadata:dict,):
        """

        Args:
            local_configuration:
            remote_configuration:

        Returns:

        """



        kwargs = dict(hash_id=self.remote_table_hashed_name,
                      build_configuration=remote_configuration, )


        local_metadata_kwargs = dict(local_hash_id=self.local_hash_id,
                               build_configuration=local_configuration,
                               remote_table__hash_id=self.remote_table_hashed_name)


        TimeSerieNode.patch_build_configuration(remote_table_patch=kwargs,
                                                data_source_id=self.data_source.id,
                                                build_meta_data=remote_build_metadata,
                                                local_table_patch=local_metadata_kwargs)

    def local_persist_exist_set_config(self, local_configuration:dict, remote_configuration:dict,data_source:dict,
                                       time_serie_source_code_git_hash:str, time_serie_source_code:str,
        remote_build_metadata:dict,
                                       ):
        """
        This method runs on initialization of the TimeSerie class. We also use it to retrieve the table if
        is already persisted
        :param config:

        :return:
        """

        remote_build_configuration=None
        if hasattr(self, "remote_build_configuration"):
            remote_build_configuration = self.remote_build_configuration
        if hasattr(self, "remote_build_metadata"):
            remote_build_metadata = self.remote_build_metadata
        remote_table_exist = True
        if remote_build_configuration is None:
            #create remote table
            remote_table_exist = False
            try:

                # table may not exist but
                remote_build_metadata = remote_configuration["build_meta_data"] if "build_meta_data" in remote_configuration.keys() else {}
                remote_configuration.pop("build_meta_data", None)
                kwargs = dict(hash_id=self.remote_table_hashed_name,
                              time_serie_source_code_git_hash=time_serie_source_code_git_hash,
                              time_serie_source_code=time_serie_source_code,
                              build_configuration=remote_configuration,
                              data_source=data_source.model_dump(),
                              build_meta_data=remote_build_metadata)
                if self.human_readable is not None:
                    kwargs["human_readable"] = self.human_readable
                # node_kwargs={"hash_id":self.remote_table_hashed_name,
                #                                                  "source_class_name":self.class_name,
                #                                                  "human_readable": self.human_readable,
                #
                #                                                  }


                # kwargs["source_class_name"]=self.class_name
                self.metadata = self.dth.create(metadata_kwargs=kwargs)

                #todo: after creating metadata always delete local parquet manager even if not exist
                # self.delete_local_parquet()

            except Exception as e:
                self.logger.exception(f"{self.remote_table_hashed_name} Could not set meta data in DB for P")
                raise e
        # check if link to local update exists

        local_table_exist = self._verify_local_ts_exists(local_configuration=local_configuration)

        return remote_table_exist,local_table_exist

    def _verify_local_ts_exists(self,local_configuration:Union[dict,None]=None):
        """
        Verifies that the local time serie exist in ORM
        Parameters
        ----------
        local_configuration

        Returns
        -------

        """
        local_table_exist=True
        local_build_configuration = None
        if hasattr(self, "local_build_configuration"):
            local_build_configuration, local_build_metadata = self.local_build_configuration, self.local_build_metadata
        if local_build_configuration is None:
            local_table_exist=False
            local_update = TimeSerieLocalUpdate.filter(local_hash_id=self.local_hash_id,
                                                       remote_table__data_source__id=self.data_source.id)
            if len(local_update) == 0:
                local_build_metadata = local_configuration[
                    "build_meta_data"] if "build_meta_data" in local_configuration.keys() else {}
                local_configuration.pop("build_meta_data", None)
                metadata_kwargs=dict(local_hash_id=self.local_hash_id,
                              build_configuration=local_configuration,
                              remote_table__hash_id=self.metadata['hash_id'],
                                     description=self.description,
                                     data_source_id=self.data_source.id
                                     )
                if self.human_readable is not None:
                    metadata_kwargs["human_readable"] = self.human_readable

                node_kwargs = {"hash_id": self.local_hash_id,
                               "source_class_name": self.class_name,
                               "human_readable": self.human_readable,
                               "data_source_id" : self.data_source.id
                               }

                local_metadata = TimeSerieLocalUpdate.create(metadata_kwargs=metadata_kwargs,
                                                             node_kwargs=node_kwargs,

                                              )
                self.local_build_configuration = local_metadata["build_configuration"]
                self.local_build_metadata = local_metadata["build_meta_data"]
                self.local_metadata = local_metadata
            else:
                local_metadata=local_update
            self.local_metadata=local_metadata
        return   local_table_exist
    def _verify_insertion_format(self,temp_df):
        """
        verifies that data frame is properly configured
        Parameters
        ----------
        temp_df :

        Returns
        -------

        """
        if self.remote_table_hashed_name!=self.local_hash_id:

            assert temp_df.index.names==["time_index","asset_symbol"] or  temp_df.index.names==["time_index","asset_symbol","execution_venue_symbol"]
        if isinstance(temp_df.index,pd.MultiIndex)==False:
            # assert temp_df.index.name is not None
            pass

    def update_details_exist(self):
        """

        Returns
        -------

        """
        exist= TimeSerieLocalUpdate.update_details_exist(local_metadata=self.local_metadata)
        return exist
    def build_update_details(self,source_class_name):
        """

        Returns
        -------

        """

        update_kwargs=dict(source_class_name=source_class_name,
                           local_metadata=self.local_metadata,
                           )


        metadatas=self.dth.build_or_update_update_details(metadata=self.metadata,
                                                **update_kwargs)

        self.metadata = metadatas["metadata"]
        self.local_metadata = metadatas["local_metadata"]

    def patch_update_details(self,local_hash_id=None,
                             **kwargs):
        """
        Patch update details ofr related_table
        Parameters
        ----------
        hash_id :
        kwargs :

        Returns
        -------

        """
        if local_hash_id is not None:
            kwargs["use_local_hash_id"]=local_hash_id
            metadata=self.dth.build_or_update_update_details(metadata=self.metadata,**kwargs)
            return metadata
        kwargs["local_metadata"]=self.local_metadata
        metadatas=self.dth.build_or_update_update_details(metadata=self.metadata,**kwargs)
        self.metadata=metadatas["metadata"]
        self.local_metadata=metadatas["local_metadata"]

    def patch_table(self,**kwargs):
        self.dth.patch(metadata=self.metadata, **kwargs)

    def protect_from_deletion(self,protect_from_deletion=True):
        self.dth.patch(metadata=self.metadata, protect_from_deletion=protect_from_deletion)

    def open_for_everyone(self,open_for_everyone=True):
        self.dth.patch(metadata=self.metadata, open_for_everyone=open_for_everyone)

    def set_start_of_execution(self,**kwargs):
        return self.dth.set_start_of_execution(metadata=self.metadata,**kwargs)
    def set_end_of_execution(self,**kwargs):
        return self.dth.set_end_of_execution(metadata=self.metadata, **kwargs)
    def reset_dependencies_states(self,hash_id_list):
        return self.dth.reset_dependencies_states(metadata=self.metadata, hash_id_list=hash_id_list)
    def get_pending_nodes(self,table_id_list:list, filter_by_update_time:bool ):

        """

        Parameters
        ----------
        filter_by_update_time :

        Returns
        -------

        """
        request_kwargs=dict(table_id_list=table_id_list,   filter_by_update_time=filter_by_update_time)
        data=self.dth.get_pending_nodes(metadata=self.metadata,**request_kwargs)
        return data["dependecies_updated"],data['pending_nodes'],data["next_rebalances"],data["error_on_dependencies"]


    #table dependes

    def get_latest_value(self, asset_symbols:list) -> [datetime.datetime,Dict[str, datetime.datetime]]:

        metadata= self.dth.get(hash_id=self.remote_table_hashed_name,data_source__id=self.data_source.id)

        last_index_value,last_multiindex=None,None
        if "sourcetableconfiguration" in metadata.keys():
            if metadata['sourcetableconfiguration'] is not None:
                last_index_value=metadata['sourcetableconfiguration']['last_time_index_value']
                if last_index_value is None:
                    return last_index_value,last_multiindex
                last_index_value=self.dth.request_to_datetime(last_index_value)
                if metadata['sourcetableconfiguration']['multi_index_stats'] is not None:
                    last_multiindex=metadata['sourcetableconfiguration']['multi_index_stats']['max_per_asset_symbol']
                    if last_multiindex is not None:
                        last_multiindex={symbol:{ev:self.dth.request_to_datetime(v) for ev,v in ev_dict.items()} for symbol,ev_dict in last_multiindex.items()}

        if asset_symbols is not None and last_multiindex is not None:
            last_multiindex = {asset: value for asset, value in last_multiindex.items() if asset in asset_symbols}

        return last_index_value,last_multiindex

    def _add_to_data_source(self,data_df,overwrite:bool):
        raise NotImplementedError

    def persist_updated_data(self, temp_df: pd.DataFrame,historical_update_id:Union[int,None],
                             update_tracker: Union[object, None] = None,
                             overwrite=False):
        """
        Main update time series function, it is called from TimeSeries class
        Parameters
        ----------
        temp_df
        latest_value
        session

        Returns
        -------

        """

        self.local_metadata = self.dth.upsert_data_into_table(
            metadata=self.metadata,
            local_metadata=self.local_metadata,
            data=temp_df,
            historical_update_id=historical_update_id,
            overwrite=overwrite,data_source=self.data_source,
        logger=self.logger
        )

        if update_tracker is not None:
            update_tracker.set_end_of_execution(hash_id=self.local_hash_id, error_on_update=False)

class TimeScaleLocalPersistManager(PersistManager):
    """
    Main Controler to interacti with TimeSerie ORM
    """





    def get_persisted_ts(self):
        """
        full Request of the persisted data should always default to DB
        :return:
        """

        persisted_df = self.dth.get_data_by_time_index(metadata=self.metadata)

        return persisted_df

    def get_df_greater_than_in_table(self, target_value: datetime.datetime, great_or_equal: bool,

                            symbol_list:Union[list,None]=None,
                            columns: Union[list, None] = None
                            ):
        """

        Parameters
        ----------
        target_value
        great_or_equal

        Returns
        -------

        """



        filtered_data = self.dth.get_data_by_time_index(start_date=target_value, metadata=self.metadata,
                                                        connection_config=self.get_data_source_connection_details(),
                                                        columns=columns,
                                                        asset_symbols=symbol_list,
                                                        great_or_equal=great_or_equal, )


        return filtered_data

    def filter_by_assets_ranges(self, asset_ranges_map: dict, force_db_look: bool):

        if force_db_look:
            if  self.metadata["sourcetableconfiguration"] is not None:
                assert "asset_symbol" in self.metadata["sourcetableconfiguration"]["index_names"],"Table does not contain asset_symbol column"
            connection_config=DynamicTableDataSource.get_data_source_connection_details(self.metadata["data_source"]["id"])

            df=self.dth.filter_by_assets_ranges(table_name=self.metadata['hash_id'],asset_ranges_map=asset_ranges_map,

                                                connection_config=connection_config)
            df["time_index"]=pd.to_datetime(df["time_index"])
            df=df.set_index(self.metadata["sourcetableconfiguration"]["index_names"])
            
            
        else:
            raise NotImplementedError
        return df
    def get_df_between_dates(self, start_date, end_date, great_or_equal=True,
                                      less_or_equal=True,
                                      asset_symbols: Union[list, None] = None,
                                      columns: Union[list, None] = None):
        return self._get_df_between_dates_from_db(start_date, end_date, great_or_equal=great_or_equal,
                                      less_or_equal=less_or_equal,
                                      asset_symbols=asset_symbols,
                                      columns = columns)



    def _get_df_between_dates_from_db(self, start_date, end_date,  great_or_equal=True,
                                      less_or_equal=True,
                                      asset_symbols: Union[list, None] = None,
                                      columns: Union[list, None] = None
                                      ):
        """

        Parameters
        ----------
        start_date
        end_date

        Returns
        -------

        """



        # if start date is no after earlier local retetion default ot DB
        if "id" not in self.metadata.keys():
            raise Exception(f"No id in f{self.metadata}")
        filtered_data = self.dth.get_data_by_time_index(metadata=self.metadata, start_date=start_date,
                                                        end_date=end_date, great_or_equal=great_or_equal,
                                                        less_or_equal=less_or_equal,
                                                        asset_symbols=asset_symbols,
                                                        columns=columns,connection_config=self.get_data_source_connection_details()
                                                        )
        self.logger.warning(
            f"Data is not been pulled from local storage, review  storage policy to improve performace {start_date} - {end_date}")
        return filtered_data







       


    def upsert_data(self, data_df: pd.DataFrame):

        self.add_data_to_timescale(temp_df=data_df, overwrite=True)


    def get_earliest_value(self) -> datetime.datetime:
        earliest_value = self.dth.get_earliest_value(hash_id=self.remote_table_hashed_name)
        return earliest_value

    def get_full_source_data(self, engine="pandas"):
        """
        Returns full stored data, uses multiprocessing to achieve several queries by rows and speed
        :return:
        """

        from joblib import Parallel, delayed
        from tqdm import tqdm

        metadata = self.dth.get_configuration(hash_id=self.remote_table_hashed_name)
        earliest_obs = metadata["sourcetableconfiguration"]["last_time_index_value"]
        latest_value = metadata["sourcetableconfiguration"]["earliest_index_value"]

        ranges = list(pd.date_range(earliest_obs, latest_value, freq="1 m"))

        if earliest_obs not in ranges:
            ranges = [earliest_obs] + ranges

        if latest_value not in ranges:
            ranges.append(latest_value)

        def get_data(ranges, i, metadata, dth):

            s, e = ranges[i], ranges[i + 1]
            tmp_data = dth.get_data_by_time_index(start_date=s, end_date=e, metadata=self.metadata,
                                                  great_or_equal=True, less_or_equal=False)

            tmp_data = tmp_data.reset_index()
            return tmp_data

        dfs = Parallel(n_jobs=10)(
            delayed(get_data)(ranges, i, self.remote_table_hashed_name, engine) for i in tqdm(range(len(ranges) - 1)))
        dfs = pd.concat(dfs, axis=0)
        dfs = dfs.set_index(self.metadata["table_config"]["index_names"])
        return dfs


    def set_policy(self, interval: str,overwrite:bool,comp_type:str):
        if self.metadata is not None:
            retention_policy_config = self.metadata["retention_policy_config"]
            compression_policy_config=self.metadata["compression_policy_config"]
            if comp_type =="retention":
                if retention_policy_config is None or overwrite == True:
                    status = self.dth.set_retention_policy(interval=interval, metadata=self.metadata)
            if comp_type=="compression":
                if compression_policy_config is None or overwrite == True:
                    status = self.dth.set_compression_policy(interval=interval, metadata=self.metadata)
        else:
            self.logger.warning("Retention policy couldnt be set as TS is not yet persisted")
    def set_policy_for_descendants(self,policy,comp_type:str,exclude_ids:Union[list,None]=None,extend_to_classes=False):
        self.dth.set_policy_for_descendants(hash_id=self.remote_table_hashed_name,pol_type=comp_type,policy=policy,
                                            exclude_ids=exclude_ids,extend_to_classes=extend_to_classes)



    def delete_after_date(self, after_date: str):
        self.dth.delete_after_date(metadata=self.metadata, after_date=after_date)





class DataLakePersistManager(PersistManager):
    """
    A class to manage data persistence in a local data lake.

    This class handles the storage and retrieval of time series data in a local file system,
    organized by date ranges and table hashes.
    """
    TIME_PARTITION = "TIME_PARTITION"

    def __init__(self, *args,**kwargs):
        """
        Initializes the DataLakePersistManager with configuration from environment variables.
        """


        super().__init__(*args,**kwargs)
        self.already_introspected = self.set_introspection(introspection=False)



    def set_introspection(self, introspection: bool):
        """
        This methos is critical as it control the level of introspection and avouids recursivity\
        This happens for example when TimeSeries.update_series_from_source(*,**):
        TimeSeries.update_series_from_source(latest_value,*,**):
            self.get_latest_value() <- will incurr in a circular refefence using local data late
        Args:
            introspection:

        Returns:

        """
        self.logger.debug(f"Setting introspection for {self.local_hash_id}")
        self.introspection = introspection




    def get_latest_value_in_table(self, ts, asset_symbols: list, *args, **kwargs) -> [datetime.datetime,
                                                                             Dict[str, datetime.datetime]]:
        """
        Returns: [datetime.datetime,Dict[str, datetime]]
        """
        if self.introspection:
            return None, None

        file_path = self.get_file_path()
        last_index_value, last_multiindex = None, None
        if not os.path.exists(file_path):
            # create the object
            _ = self._query_datalake(
                ts=ts,
                symbol_list=asset_symbols,
                great_or_equal=True,
                start_date=None,
                *args, **kwargs
            )

        last_index_value, last_multiindex = self._get_parquet_latest_value()
        return last_index_value, last_multiindex

    def get_df_greater_than_in_table(self, ts: object,
                                     latest_value: pd.Timestamp,
                                     symbol_list: List[str],
                                     great_or_equal: bool,
                                     *args,
                                     **kwargs):
        data = self._query_datalake(
            ts=ts,
            start_date=latest_value,
            symbol_list=symbol_list,
            great_or_equal=great_or_equal,
            *args,
            **kwargs
        )
        return data

    def get_df_between_dates(self, ts, symbol_list: List[str],
                             great_or_equal: bool, less_or_equal: bool,
                             end_date: datetime.datetime, start_date: datetime.datetime,
                             ):
        return self._query_datalake(ts=ts, symbol_list=symbol_list, great_or_equal=great_or_equal,
                                    less_or_equal=less_or_equal, start_date=start_date, end_date=end_date)










