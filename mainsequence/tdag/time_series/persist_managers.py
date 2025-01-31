import pandas as pd
import datetime
from typing import Union, List,Dict
import os
from mainsequence.logconf import logger


from mainsequence.tdag_client import (LocalTimeSerie,
                                      LocalTimeSeriesDoesNotExist, PodLocalLake,
                                      DynamicTableDoesNotExist, DynamicTableDataSource, CONSTANTS, DynamicTableMetaData,
                                      DataUpdates)

from mainsequence.tdag_client.models import BACKEND_DETACHED, none_if_backend_detached, DynamicTableHelpers
import json

class APIPersistManager:

    def __init__(self,data_source_id:int,local_hash_id:str):
        self.data_source_id = data_source_id
        self.local_hash_id = local_hash_id

        local_metadata = LocalTimeSerie.get(local_hash_id=self.local_hash_id,
                                                  remote_table__data_source__id=self.data_source_id
                                                  )
        self.local_metadata=local_metadata


    def get_df_between_dates(self, start_date, end_date, great_or_equal=True,
                             less_or_equal=True,
                             unique_identifier_list: Union[list, None] = None,
                             columns: Union[list, None] = None,
                             unique_identifier_range_map: Union[dict, None] = None,):
        filtered_data = self.local_metadata.get_data_between_dates_from_api(

                                                        start_date=start_date,
                                                        end_date=end_date, great_or_equal=great_or_equal,
                                                        less_or_equal=less_or_equal,
                                                        unique_identifier_list=unique_identifier_list,
                                                        columns=columns,
                                                        unique_identifier_range_map=unique_identifier_range_map)

        if len(filtered_data) == 0:
            logger.info(f"Data from {self.local_hash_id} is empty in request ")
            return filtered_data

        #fix types

        stc = self.local_metadata.remote_table.sourcetableconfiguration
        filtered_data[stc.time_index_name] = pd.to_datetime(filtered_data[stc.time_index_name])
        for c, c_type in stc.column_dtypes_map.items():
            if c!=stc.time_index_name:
                if c_type=="object":
                    c_type="str"
                filtered_data[c]=filtered_data[c].astype(c_type)
        filtered_data=filtered_data.set_index(stc.index_names)
        return filtered_data

    def filter_by_assets_ranges(self, unique_identifier_range_map: dict,time_serie:"TimeSerie"):
        df = self.get_df_between_dates(start_date=None, end_date=None, unique_identifier_range_map=unique_identifier_range_map)
        return df

class PersistManager:
    def __init__(self,
                 data_source,
                 local_hash_id: int,

                 description: Union[str, None] = None,
                 class_name: Union[str, None] = None,
                 human_readable: Union[str, None] = None, metadata: Union[dict, None] = None,
                 local_metadata: Union[dict, None] = None

                 ):
        self.data_source = data_source
        self.local_hash_id = local_hash_id

        if local_metadata is not None and metadata is None:
            # query remote hash_id
            metadata = local_metadata.remote_table
        self.description=description
        self.logger = logger


        self.table_model_loaded = False
        self.human_readable = human_readable if human_readable is not None else local_hash_id

        self.class_name = class_name
        if self.local_hash_id is not None:
            self.synchronize_metadata(meta_data=metadata,
                                      local_metadata=local_metadata,
                                      class_name=class_name)

    @classmethod
    def get_from_data_type(self,data_source:DynamicTableDataSource,*args, **kwargs):

        data_type = data_source.related_resource_class_type
        if data_type in CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE:
            return DataLakePersistManager(data_source=data_source,
                                         *args, **kwargs)

        elif data_type in CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB:
            return TimeScaleLocalPersistManager(data_source=data_source, *args, **kwargs)

    def synchronize_metadata(self, meta_data: Union[dict, None], local_metadata: Union[dict, None],
                             set_last_index_value: bool = False,
                             class_name: Union[str, None] = None
                             ):
        """
        forces a synchronization between table and metadata
        :return:
        """
        if BACKEND_DETACHED():
            self.local_metadata=local_metadata
            return None

        # start with remote metadata
        if set_last_index_value == True:
            TimeSerieLocalUpdate.set_last_update_index_time(metadata=self.local_metadata)
        if meta_data is None or local_metadata is None:  # avoid calling 2 times the DB
            meta_data = {}

            local_metadata = {}  # set to empty in case not exist
            local_metadata = LocalTimeSerie.get(local_hash_id=self.local_hash_id,
                                                remote_table__data_source__id=self.data_source.id
                                                      )


        if local_metadata is not None:
            self.local_build_configuration = local_metadata.build_configuration
            self.local_build_metadata = local_metadata.build_meta_data
            self.local_metadata = local_metadata

            # metadata should always exist
            self.metadata = local_metadata.remote_table



    def depends_on_connect(self,new_ts:"TimeSerie",is_api:bool):
        """
        Connects a time Serie as relationship in the DB
        Parameters
        ----------
        new_ts :

        Returns
        -------

        """

        if is_api ==False:
            try:
                human_readable = new_ts.local_persist_manager.metadata.human_readable
            except KeyError:
                human_readable = new_ts.human_readable
            self.local_metadata.depends_on_connect(

                                        source_local_hash_id=self.local_metadata.local_hash_id,
                                        target_local_hash_id=new_ts.local_hash_id,

                                        target_class_name=new_ts.__class__.__name__,
                                        target_human_readable=human_readable,

                                        source_data_source_id=self.data_source.id,
                                        target_data_source_id=new_ts.data_source.id

                                        )
        else:
            self.local_metadata.depends_on_connect_remote_table(
                source_hash_id=self.metadata.hash_id,
                source_local_hash_id=self.local_metadata.local_hash_id,
                source_data_source_id=self.data_source.id,

                                        target_data_source_id=new_ts.data_source_id,
                                        target_local_hash_id=new_ts.local_hash_id
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
        depth_df = self.local_metadata.get_all_dependencies_update_priority()
        return depth_df

    def set_ogm_dependencies_linked(self):

        self.local_metadata.set_ogm_dependencies_linked()

    @property
    def update_details(self):

        return self.local_metadata.localtimeserieupdatedetails

    @property
    def run_configuration(self):

        return self.local_metadata.run_configuration


    @property
    def source_table_configuration(self):
        if "sourcetableconfiguration" in self.metadata.keys():
            return self.metadata['sourcetableconfiguration']
        return None
    @none_if_backend_detached
    def update_source_informmation(self, git_hash_id:str, source_code:str):
        """

        Args:
            git_hash_id:
            source_code:

        Returns:

        """

        self.metadata = self.metadata.patch( time_serie_source_code_git_hash=git_hash_id,
                            time_serie_source_code=source_code,)

    

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


    def add_tags(self, tags: list):
        if any([t not in self.local_metadata.tags for t in tags]) == True:
            self.local_metadata.add_tags(tags=tags)

    def destroy(self, delete_only_table: bool):
        self.dth.destroy(metadata=self.metadata, delete_only_table=delete_only_table)

    @property
    def persist_size(self):
        try:
            return self.metadata['table_size']
        except KeyError:
            return 0

    def time_serie_exist(self):
        if hasattr(self,"metadata"):
            return True
        return False



    def patch_build_configuration(self,local_configuration:dict,remote_configuration:dict,
                                  remote_build_metadata:dict,):
        """

        Args:
            local_configuration:
            remote_configuration:

        Returns:

        """



        kwargs = dict(
                      build_configuration=remote_configuration, )


        local_metadata_kwargs = dict(local_hash_id=self.local_hash_id,
                               build_configuration=local_configuration,
                              )


        self.local_metadata=DynamicTableMetaData.patch_build_configuration(remote_table_patch=kwargs,
                                                data_source_id=self.data_source.id,
                                                build_meta_data=remote_build_metadata,
                                                local_table_patch=local_metadata_kwargs)

    def local_persist_exist_set_config(self,remote_table_hashed_name:str,
                                       local_configuration:dict, remote_configuration:dict,data_source:dict,
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
                kwargs = dict(hash_id=remote_table_hashed_name,
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
                self.metadata = DynamicTableMetaData.create(metadata_kwargs=kwargs)

                #todo: after creating metadata always delete local parquet manager even if not exist
                # self.delete_local_parquet()

            except Exception as e:
                self.logger.exception(f"{remote_table_hashed_name} Could not set meta data in DB for P")
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
            local_update = LocalTimeSerie.get(local_hash_id=self.local_hash_id,
                                                       remote_table__data_source__id=self.data_source.id)
            if local_update is None:
                local_build_metadata = local_configuration[
                    "build_meta_data"] if "build_meta_data" in local_configuration.keys() else {}
                local_configuration.pop("build_meta_data", None)
                metadata_kwargs=dict(local_hash_id=self.local_hash_id,
                              build_configuration=local_configuration,
                              remote_table__hash_id=self.metadata.hash_id,
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

                local_metadata = LocalTimeSerie.create(metadata_kwargs=metadata_kwargs,
                                                             node_kwargs=node_kwargs,

                                              )
                self.local_build_configuration = local_metadata.build_configuration
                self.local_build_metadata = local_metadata.build_meta_data
                self.local_metadata = local_metadata
            else:
                local_metadata=local_update
            self.local_metadata=local_metadata
            self.local_build_configuration = local_metadata.build_configuration
            self.local_build_metadata = local_metadata.build_meta_data

            # metadata should always exist
            self.metadata = local_metadata.remote_table

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

        if isinstance(temp_df.index,pd.MultiIndex)==True:
            assert temp_df.index.names==["time_index","asset_symbol"] or  temp_df.index.names==["time_index","asset_symbol","execution_venue_symbol"]

    def build_update_details(self,source_class_name):
        """

        Returns
        -------

        """

        update_kwargs=dict(source_class_name=source_class_name,
                           local_metadata=json.loads(self.local_metadata.model_dump_json())
                           )


        metadata,local_metadata=self.local_metadata.remote_table.build_or_update_update_details( **update_kwargs)

        self.metadata = metadata
        self.local_metadata = local_metadata

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
        self.metadata.patch( **kwargs)

    def protect_from_deletion(self,protect_from_deletion=True):
        self.metadata.patch( protect_from_deletion=protect_from_deletion)

    def open_for_everyone(self,open_for_everyone=True):
        self.metadata.patch(open_for_everyone=open_for_everyone)

    def set_start_of_execution(self,**kwargs):
        return self.dth.set_start_of_execution(metadata=self.metadata,**kwargs)
    def set_end_of_execution(self,**kwargs):
        return self.dth.set_end_of_execution(metadata=self.metadata, **kwargs)
    def reset_dependencies_states(self,hash_id_list):
        return self.dth.reset_dependencies_states(metadata=self.metadata, hash_id_list=hash_id_list)

    #table dependes

    def get_update_statistics(self, asset_symbols:list,
                              remote_table_hash_id,time_serie,
                              ) -> [datetime.datetime,Dict[str, datetime.datetime]]:
        if BACKEND_DETACHED(): #todo this can be optimized by running stats per data lake
            return self._get_local_lake_update_statistics(remote_table_hash_id=remote_table_hash_id,
                                                          time_serie=time_serie)

        metadata = self.local_metadata.remote_table

        last_update_in_table, last_update_per_asset = None, None

        if metadata.sourcetableconfiguration is not None:
            last_update_in_table = metadata.sourcetableconfiguration.last_time_index_value
            if last_update_in_table is None:
                return last_update_in_table, last_update_per_asset
            if metadata.sourcetableconfiguration.multi_index_stats is not None:
                last_update_per_asset = metadata.sourcetableconfiguration.multi_index_stats['max_per_asset_symbol']
                if last_update_per_asset is not None:
                    last_update_per_asset = {unique_identifier: DynamicTableHelpers.request_to_datetime(v) for unique_identifier, v in last_update_per_asset.items()}

        if asset_symbols is not None and last_update_per_asset is not None:
            last_update_per_asset = {asset: value for asset, value in last_update_per_asset.items() if asset in asset_symbols}

        return last_update_in_table, last_update_per_asset

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

        self.local_metadata = DynamicTableHelpers.upsert_data_into_table(
            local_metadata=self.local_metadata,
            data=temp_df,
            data_source=self.data_source,

        )



    def get_persisted_ts(self):
        """
        full Request of the persisted data should always default to DB
        :return:
        """

        persisted_df = self.dth.get_data_by_time_index(metadata=self.metadata)

        return persisted_df

    def filter_by_assets_ranges(self, asset_ranges_map: dict,time_serie):
        if BACKEND_DETACHED == False:
            if self.metadata["sourcetableconfiguration"] is not None:
                assert "asset_symbol" in self.metadata["sourcetableconfiguration"][
                    "index_names"], "Table does not contain asset_symbol column"
        else:
            if isinstance(self, DataLakePersistManager):
                self.verify_if_already_run(time_serie)
        df = self.dth.filter_by_assets_ranges(metadata=self.metadata, asset_ranges_map=asset_ranges_map,
                                              data_source=self.data_source, local_hash_id=time_serie.local_hash_id)


        return df

    def get_earliest_value(self,remote_table_hash_id) -> datetime.datetime:
        earliest_value = self.dth.get_earliest_value(hash_id=remote_table_hash_id)
        return earliest_value

    def get_df_between_dates(self, start_date, end_date, great_or_equal=True,
                             less_or_equal=True,
                             unique_identifier_list: Union[list, None] = None,
                             columns: Union[list, None] = None):

        filtered_data = self.data_source.get_data_by_time_index(local_metadata=self.local_metadata,
                                                        start_date=start_date,
                                                        end_date=end_date,
                                                        great_or_equal=great_or_equal,
                                                        less_or_equal=less_or_equal,
                                                        unique_identifier_list=unique_identifier_list,
                                                        columns=columns,
                                                        )

        return filtered_data


class TimeScaleLocalPersistManager(PersistManager):
    """
    Main Controler to interacti with TimeSerie ORM
    """







    def get_full_source_data(self,remote_table_hash_id, engine="pandas"):
        """
        Returns full stored data, uses multiprocessing to achieve several queries by rows and speed
        :return:
        """

        from joblib import Parallel, delayed
        from tqdm import tqdm

        metadata = self.dth.get_configuration(hash_id=remote_table_hash_id)
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
            delayed(get_data)(ranges, i, remote_table_hash_id, engine) for i in tqdm(range(len(ranges) - 1)))
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
    def set_policy_for_descendants(self,remote_table_hash_id:str,
                                   policy,comp_type:str,exclude_ids:Union[list,None]=None,extend_to_classes=False):
        self.dth.set_policy_for_descendants(hash_id=remote_table_hash_id,pol_type=comp_type,policy=policy,
                                            exclude_ids=exclude_ids,extend_to_classes=extend_to_classes)



    def delete_after_date(self, after_date: str):
        self.dth.delete_after_date(metadata=self.metadata, after_date=after_date)

    def get_table_schema(self,table_name):
        return self.metadata["sourcetableconfiguration"]["column_dtypes_map"]





class DataLakePersistManager(PersistManager):

    """
    A class to manage data persistence in a local data lake.

    This class handles the storage and retrieval of time series data in a local file system,
    organized by date ranges and table hashes.
    """

    def __init__(self, *args,**kwargs):
        """
        Initializes the DataLakePersistManager with configuration from environment variables.
        """
        super().__init__(*args,**kwargs)
        self.set_already_run(already_run=False)
        self.set_is_introspecting(False)


    def set_is_introspecting(self,is_introspecting):
        self.is_introspecting = is_introspecting

    def verify_if_already_run(self,ts):
        """
        This method handles all the configuration and setup necessary when running a detached local data lake
        :param ts:
        :return:
        """
        from mainsequence.tdag_client.models import BACKEND_DETACHED
        from mainsequence.tdag.time_series import WrapperTimeSerie
        if self.already_run== True or self.is_introspecting ==True:
            return None
        update_statistics=DataUpdates(update_statistics=None,max_time_index_value=None)
        if BACKEND_DETACHED() and self.data_source.related_resource_class_type in CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE:
            self.set_is_introspecting(True)
            self.metadata = {"sourcetableconfiguration":None, "hash_id": ts.remote_table_hashed_name,
                            "table_name":ts.remote_table_hashed_name
                            }
            self.local_metadata= {"local_hash_id":self.local_hash_id,"remote_table":self.metadata}
            last_update_in_table=None
            if self.table_exist(table_name=ts.remote_table_hashed_name):
                # check if table is complete and continue with earliest latest value to avoid data gaps
                last_update_in_table, last_update_per_unique_identifier = ts.get_update_statistics()
                if last_update_per_unique_identifier is not None:
                    last_update_in_table = ts.get_earliest_updated_asset_filter(last_update_per_unique_identifier=last_update_per_unique_identifier,
                                                                                unique_identifier_list=None)

                update_statistics.update_statistics=last_update_per_unique_identifier
                update_statistics.max_time_index_value=last_update_in_table


            self.logger.debug(f"Building local data lake from latest value  {last_update_in_table}")

            if isinstance(ts,WrapperTimeSerie):
                df = None
                for _,sub_ts in ts.related_time_series.items():
                    sub_ts.local_persist_manager #query the first run
            else:
                df = ts.update_series_from_source(update_statistics=update_statistics)


            if df is None:
                return None
            if df.shape[0] == 0:
                return None
            self.dth.upsert_data_into_table(metadata={"table_name": ts.remote_table_hashed_name}, local_metadata=None, data=df,
                                            logger=self.logger, overwrite=True, historical_update_id=None,
                                            data_source=self.data_source)
            #verify pickle exist
            ts.persist_to_pickle()
            self.set_already_run(True)  # set before the update to stop recurisivity



    def set_already_run(self, already_run: bool):
        """
        This methos is critical as it control the level of introspection and avouids recursivity\
        This happens for example when TimeSeries.update_series_from_source(*,**):
        TimeSeries.update_series_from_source(latest_value,*,**):
            self.get_update_statistics() <- will incurr in a circular refefence using local data late
        Args:
            introspection:

        Returns:

        """

        self.already_run = already_run

    def _get_local_lake_update_statistics(self,remote_table_hash_id,time_serie):
        from mainsequence.tdag_client.data_sources_interfaces.local_data_lake import DataLakeInterface
        assert self.data_source.related_resource_class_type in CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE
        if self.already_run == False:
            self.verify_if_already_run(time_serie)
        last_index_value, last_multiindex = DataLakeInterface(
            data_lake_source=self.data_source,
        ).get_parquet_latest_value(
            table_name=remote_table_hash_id
        )
        if last_multiindex is not None:
            if len(last_multiindex)==0:
                last_multiindex=None
        return last_index_value, last_multiindex

    def table_exist(self,table_name):
        from mainsequence.tdag_client.data_sources_interfaces.local_data_lake import DataLakeInterface
        assert self.data_source.related_resource_class_type in CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE
        return DataLakeInterface(data_lake_source=self.data_source,
                                                              ).table_exist(
            table_name=table_name)
    def get_table_schema(self,table_name):
        from mainsequence.tdag_client.data_sources_interfaces.local_data_lake import DataLakeInterface
        dli=DataLakeInterface(data_lake_source=self.data_source,
                         logger=self.logger)
        schema=dli.get_table_schema(table_name=table_name)
        schema = {field.name: field.type for field in schema}
        return schema

    def get_df_between_dates(self,time_serie:"TimeSerie", *args,**kwargs):
        if self.already_run ==False:
            self.verify_if_already_run(time_serie)
        filtered_data=super().get_df_between_dates( *args,**kwargs)

        return filtered_data