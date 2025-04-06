import fire

import json

import runpy
import tempfile
import os
from pathlib import Path

import requests
import yaml
from requests.structures import CaseInsensitiveDict


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

def run_app(app_name, configuration):
    from mainsequence.virtualfundbuilder.resource_factory.app_factory import APP_REGISTRY
    app = APP_REGISTRY[app_name]

    configuration_json = yaml.load(configuration, Loader=yaml.UnsafeLoader)
    configuration_pydantic = app.configuration_class(**configuration_json)
    results = app(configuration_pydantic).run()
    print(f"Finished App {app_name} run with results: {results}")

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

def get_py_modules(folder_path):
    if not os.path.isdir(folder_path): return []
    files = os.listdir(folder_path)
    files = [f for f in files if f[0] not in ["_", "."] and f.endswith(".py")]
    return [f.split(".")[0] for f in files]

def get_pod_configuration():
    print("Get pod configuration")

    project_library = os.getenv("PROJECT_LIBRARY_NAME")
    if not project_library:
        raise RuntimeError("PROJECT_LIBRARY_NAME is not set in environment")

    project_path = os.getenv("VFB_PROJECT_PATH")

    # Gather all submodules in time_series
    time_series_package = f"{project_library}.time_series"
    time_series_modules = get_py_modules(os.path.join(project_path, "time_series"))

    # Gather all submodules in rebalance_strategies
    rebalance_package = f"{project_library}.rebalance_strategies"
    rebalance_modules = get_py_modules(os.path.join(project_path, "rebalance_strategies"))

    # Build the temporary Python script to import all files
    script_lines = [
        "# -- Auto-generated imports for time_series --"
    ]

    for mod in time_series_modules:
        script_lines.append(f"import {time_series_package}.{mod}")
    script_lines.append("# -- Auto-generated imports for rebalance_strategies --")
    for mod in rebalance_modules:
        script_lines.append(f"import {rebalance_package}.{mod}")
    script_lines.append("")
    script_lines.append("from mainsequence.virtualfundbuilder.agent_interface import TDAGAgent")
    script_lines.append("print('Initialize TDAGAgent')")
    script_lines.append("tdag_agent = TDAGAgent()")

    TMP_SCRIPT = "\n".join(script_lines)

    # Write out to a temporary .py file and run
    temp_dir = tempfile.mkdtemp()
    python_file_path = Path(temp_dir) / "load_pod_configuration.py"
    with open(python_file_path, "w", encoding="utf-8") as f:
        f.write(TMP_SCRIPT)
    runpy.run_path(str(python_file_path), run_name="__main__")

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

    def __init__(self):
        from mainsequence.virtualfundbuilder.utils import get_vfb_logger
        self.logger = get_vfb_logger()

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
            elif execution_type == "system_job":
                get_pod_configuration()
            elif execution_type == "app":
                run_app(app_name=os.getenv("APP_NAME"), configuration=os.getenv("APP_CONFIGURATION"))
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

