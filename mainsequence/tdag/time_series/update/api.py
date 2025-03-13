from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from multiprocessing import Process

app = FastAPI()
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


@app.on_event("startup")
def startup():

    from mainsequence.tdag.time_series.update.scheduler import SchedulerUpdater
    from mainsequence.mainsequence_client import Scheduler

    new_scheduler = Scheduler.get(uid=app.state.scheduler_uid)

    kwargs = app.state.scheduler_kwargs
    if kwargs is None:
        kwargs = {
            'debug': False,
            'update_tree': True,
            'break_after_one_update': False,
            'wait_for_update': True,
            'raise_exception_on_error': False,
            'update_extra_kwargs': {},
            'run_head_in_main_process': False,
            'force_update': False,
            'sequential_update': False,
            'update_only_tree': False,
            "_api_port":app.state.port,

        }

    kwargs["uid"] = app.state.scheduler_uid
    p = Process(target=SchedulerUpdater.start_from_uid, kwargs=kwargs,name=new_scheduler.name)
    p.start()

    app.state_scheduler_process_pip = p.pid