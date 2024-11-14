
import fire


class TDAGApp:

    def start_scheduler(self,scheduler_id:int, port:int,host="0.0.0.0", reload=False):
        from .time_series.update.utils import start_scheduler_api
        start_scheduler_api(scheduler_id,port,host,reload)

        start_scheduler_api(cheduler_uid=scheduler_id, port = port, )

if __name__ == "__main__":
    fire.Fire(TDAGApp)
