from mainsequence.tdag.logconf import get_tdag_logger
import ray
import os

logger = get_tdag_logger()
TDAG_RAY_CLUSTER_ADDRESS = os.getenv("TDAG_RAY_CLUSTER_ADDRESS")
NAMESPACE = "time_series_update"
class RayUpdateManager:
    """
    Controller for interactions with ray cluster
    """
    def __init__(self, scheduler_uid,
                 skip_health_check=False,
                 local_mode=False):

        self.scheduler_uid = scheduler_uid

        if skip_health_check == False:
            self.is_node_healthy = self.check_node_is_healthy_in_ip()
            if self.is_node_healthy:
                self.verify_ray_is_initialized(local_mode)

    # Node health interactions

    def verify_ray_is_initialized(self, local_mode=False):
        from mainsequence import tdag
        import os
        from mainsequence.tdag.config import Configuration
        if ray.is_initialized() == False:
            self.check_node_is_healthy_in_ip()
            ray_address = TDAG_RAY_CLUSTER_ADDRESS
            env_vars = {
                "RAY_PROFILING": "0", "RAY_event_stats": "0",
                "RAY_BACKEND_LOG_LEVEL": "error",
            }

            for c in Configuration.OBLIGATORY_ENV_VARIABLES:
                env_vars[c] = os.environ.get(c)

            extra_ray_env = os.getenv("EXTRA_RAY_ENV_VARIABLES")
            if extra_ray_env:
                for env_key in extra_ray_env.split(","):
                    env_val = os.environ.get(env_key)
                    assert env_val, f"{env_key} is not set"
                    env_vars[env_key] = env_val

            kwargs = dict(address=ray_address,
                          namespace=NAMESPACE,
                          local_mode=local_mode,
                          # log_to_driver=False,
                          runtime_env={"env_vars": env_vars,
                                       "py_modules": [tdag]
                                       },
                          )  # Todo add ray cluster configuration

            ray.init(**kwargs)

    def shutdown_manager(self):
        if ray.is_initialized() == True:
            ray.shutdown()

    def check_node_is_healthy_in_ip(self) -> bool:
        return True  # todo get function out oof experimental
        healthy = False
        api_address = configuration.conf.distributed_config["ray"]["head_node_ip"]
        try:
            all_nodes = list_nodes()
        except ConnectionError:
            return False

        for n in list_nodes():
            if api_address == n["node_ip"] and n["state"] == "ALIVE":
                healthy = True
        return healthy

    # misc helpers
    def get_results_from_futures_list(self, futures: list) -> list:
        """
        should be a list of futures objects ray.remote()
        Args:
            futures ():

        Returns:

        """
        ready, unready = ray.wait(futures, num_returns=1)
        tasks_with_errors = []
        while unready:
            # logger.debug(ready)
            # logger.debug(unready)
            try:
                ray.get(ready)
            except Exception as e:
                logger.error(e)
                tasks_with_errors.append(ready)
            ready, unready = ray.wait(unready, num_returns=1)

        return tasks_with_errors

    # launch methods helpers to work with Actors
    def launch_update_task(self, kwargs_update: dict, task_options: dict):
        # update_remote_from_hash_id(**kwargs_update)
        # return  None
        future = update_remote_from_hash_id.options(**task_options).remote(**kwargs_update)
        return future

    def launch_update_task_in_process(self, kwargs_update: dict, task_options: dict):
        update_remote_from_hash_id_local(**kwargs_update)