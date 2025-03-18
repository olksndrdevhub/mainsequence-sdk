import fire

import os
import json
import importlib

from mainsequence.virtualfundbuilder.enums import RunStrategy, StrategyType
from mainsequence.virtualfundbuilder.utils import _send_strategy_to_registry, _convert_unknown_to_string
import uvicorn
from mainsequence.client.models_tdag import register_default_configuration
from mainsequence.virtualfundbuilder.utils import get_vfb_logger, get_default_documentation

import os
import runpy
import tempfile
from pathlib import Path

import requests
import sys
import logging
import logging.config
import os
from pathlib import Path

import requests
import structlog
from typing import Union

from requests.structures import CaseInsensitiveDict
from structlog.dev import ConsoleRenderer


def get_tdag_headers():
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = "Token " + os.getenv("MAINSEQUENCE_TOKEN")
    return headers

def update_job_status(status_message):
    url = f"{os.getenv('TDAG_ENDPOINT')}/orm/api/pods/job/job_run_status/"

    payload = {
        "status": status_message,
        "git_hash": os.getenv("GIT_HASH"),
    }

    response = requests.post(url, json=payload, headers=get_tdag_headers())

    if response.status_code == 200:
        data = response.json()
        print("Update success:", data)
        return data
    else:
        print("Error updating pod:", response.status_code, response.text)
        return None

def run_configuration(configuration_name):
    from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
    print("Run Timeseries Configuration")
    portfolio = PortfolioInterface.load_from_configuration(configuration_name)

    res = portfolio.run()
    print(res.head())

def run_notebook(notebook_name):
    from mainsequence.virtualfundbuilder.notebook_handling import convert_notebook_to_python_file
    print("Run Notebook")
    notebook_file_path = os.path.join(os.getenv("VFB_PROJECT_PATH"), "notebooks", f"{notebook_name}.ipynb")
    python_notebook_file = convert_notebook_to_python_file(notebook_file_path)
    runpy.run_path(python_notebook_file, run_name="__main__")

def run_script(script_name):
    print("Run script")
    python_file_path = os.path.join(os.getenv("VFB_PROJECT_PATH"), "scripts", f"{script_name}.py")
    runpy.run_path(python_file_path, run_name="__main__")

def get_pod_configuration():
    print("Get pod configuration")
    # create temporary script that imports everything and setups the agent
    TMP_SCRIPT = f"""
from mainsequence.virtualfundbuilder.agent_interface import TDAGAgent
from {os.getenv("PROJECT_LIBRARY_NAME")}.signals import *
from {os.getenv("PROJECT_LIBRARY_NAME")}.rebalance_strategies import *
tdag_agent = TDAGAgent()
"""
    temp_dir = tempfile.mkdtemp()
    python_file_path = Path(temp_dir) / "load_pod_configuration.py"

    with open(python_file_path, "w", encoding="utf-8") as f:
        f.write(TMP_SCRIPT)

    runpy.run_path(python_file_path, run_name="__main__")

def prerun_routines():
    data = update_job_status("RUNNING")
    env_update = data.get("environment_update", {})
    for key, val in env_update.items():
        os.environ[key] = val

def postrun_routines(error_on_run: bool):
    if error_on_run:
        update_job_status("FAILED")
    else:
        update_job_status("SUCCEEDED")

class VirtualFundLauncher:
    logger = get_vfb_logger()

    def run_fund_configuration_from_python(
            self,
            configuration_path_py: str, debug=True, run_strategy=RunStrategy.ALL.value

    ):
        """
        Creates fund yaml from python configuration and starts fund using the yaml path
        """
        module_name = os.path.splitext(os.path.basename(configuration_path_py))[0]
        spec = importlib.util.spec_from_file_location(module_name, configuration_path_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        portfolio_config = getattr(module, "portfolio_config")

        configuration_path = portfolio_config.build_yaml_configuration_file()
        self.run_fund_configuration_in_scheduler(configuration_path, debug=debug, run_strategy=run_strategy)


    def run_fund_configuration_in_scheduler(
            self,
            configuration_path: str,
            scheduler_name="VirtualFundScheduler",
            debug=True,
            run_strategy=RunStrategy.BACKTEST.value
    ):
        """
        Creates a portfolio interface using a yaml fund configuration, connects to VAM and runs the fund in the scheduler
        """
        from mainsequence.virtualfundbuilder.app.api.api import run_fund_configuration_in_scheduler
        run_fund_configuration_in_scheduler(configuration_path=configuration_path,scheduler_name=scheduler_name,
                                            debug=debug, run_strategy=RunStrategy(run_strategy)
                                            )

    def register_strategies(self):
        from mainsequence.virtualfundbuilder.strategy_factory.signal_factory import SignalWeightsFactory
        from mainsequence.virtualfundbuilder.strategy_factory.rebalance_factory import RebalanceFactory

        TDAG_ENDPOINT = os.getenv('TDAG_ENDPOINT', None)
        assert TDAG_ENDPOINT, "TDAG_ENDPOINT is not set"
        self.logger.debug(f"Register strategies to {TDAG_ENDPOINT}")

        subclasses = SignalWeightsFactory.get_signal_weights_strategies()
        for SignalClass in subclasses.values():
            try:
                _send_strategy_to_registry(StrategyType.SIGNAL_WEIGHTS_STRATEGY, SignalClass, is_production=True)
            except Exception as e:
                self.logger.warning("Could not register strategy to TSORM", e)

        # Register rebalance strategies
        rebalance_classes = RebalanceFactory.get_rebalance_strategies()
        for RebalanceClass in rebalance_classes.values():
            try:
                _send_strategy_to_registry(StrategyType.REBALANCE_STRATEGY, RebalanceClass, is_production=True)
            except Exception as e:
                self.logger.warning("Could mpt register strategy to TSORM", e)


    def send_default_configuration(self):
        default_config_dict = get_default_documentation()
        payload = {
            "default_config_dict": default_config_dict,
        }

        self.logger.debug(f"Send default documentation to Backend")
        payload = json.loads(json.dumps(payload, default=_convert_unknown_to_string))
        headers = {"Content-Type": "application/json"}
        try:
            response = register_default_configuration(json_payload=payload)
            if response.status_code not in [200, 201]:
                print(response.text)
        except Exception as e:
            self.logger.warning("Could register strategy to TSORM", e)


    def run_resource(self, execution_type, execution_object=None):
        error_on_run = False

        try:
            prerun_routines()
            if execution_type == "configuration":
                run_configuration(execution_object)
            elif execution_type == "script":
                run_script(execution_object)
            elif execution_type == "notebook":
                run_notebook(execution_object)
            elif execution_type == "chat_job":
                get_pod_configuration()
            else:
                raise NotImplementedError(f"Unknown execution type {execution_type}")

        except Exception as e:
            print(f"Exception during job run occured {e}")
            import traceback
            traceback.print_exc()
            error_on_run = True

        finally:
            postrun_routines(error_on_run)


if __name__ == "__main__":
    fire.Fire(VirtualFundLauncher)

