
import fire


class TDAGApp:

    def start_scheduler(self,scheduler_id:int, port:int,host="0.0.0.0", reload=False):
        from .time_series.update.utils import start_scheduler_api
        start_scheduler_api(scheduler_id,port,host,reload)

        start_scheduler_api(cheduler_uid=scheduler_id, port = port, )

    def create_indices_in_table(self,table_name:str,
                                table_index_names:dict,time_series_orm_db_connection:str):
        from mainsequence.tdag_client.utils import recreate_indexes
        from mainsequence.tdag import ogm
        from mainsequence.tdag.logconf import create_logger_in_path

        logger=create_logger_in_path(logger_name="tdag_main", application_name="tdag",
                              logger_file=f'{ogm.get_logging_path()}/{table_name}.log',
                              table_name=table_name
                              )

        logger.info(f"creating indices for table in  {table_name}")
        try:

            recreate_indexes(table_name, table_index_names, time_series_orm_db_connection,
            logger)
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == "__main__":
    fire.Fire(TDAGApp)
