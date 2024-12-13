import pytz

from mainsequence.tdag.time_series import TimeSerie

import datetime
import pandas as pd
import requests
from typing import Union
import os
def get_updates_from_mlflow(project_name:str):
    pass


class MLflowTrackingRestApi:

    ROOT_URL=os.getenv('MLFLOW_ENDPOINT')


    def get_experiment_by_name(self,experiment_name:str):
        url = self.ROOT_URL +f"/experiments/get-by-name?experiment_name={experiment_name}"
        r = requests.get(url)
        experiment = None
        if r.status_code == 200:
            experiment = r.json()["experiment"]
        return experiment
    def get_run_by_name(self,experiment_id:str,run_name:str):
        url = self.ROOT_URL + "/runs/search"
        data={"experiment_ids":[str(experiment_id)], "filter":f"run_name='{run_name}'"}
        r = requests.post(url,json=data)
        run=None
        if r.status_code == 200:
            run = r.json()["runs"][0]
  
        return run
    
    def get_all_finished_runs_df(self,experiment_name:str,include_running=False)->[pd.DataFrame,dict]:
        """
        
        Parameters
        ----------
        experiment_name : 
        include_running : 

        Returns
        -------

        """
        experiment=self.get_experiment_by_name(experiment_name=experiment_name)
        if experiment is None:
            return pd.DataFrame(),{}
        url = self.ROOT_URL + "/runs/search"
        data = {"experiment_ids": [str(experiment["experiment_id"])],}
        r = requests.post(url, json=data)
        runs = None
        if r.status_code == 200:
            if len(r.json())==0:
                return pd.DataFrame(), None
            runs = r.json()["runs"]

        comppleted_runs = [r for r in runs if
                           r["info"]["status"] == "FINISHED" and "training_completed" in pd.DataFrame(r["data"]['tags']).set_index("key").index]
        comppleted_runs=[r['info'] for r in comppleted_runs 
                         if pd.DataFrame(r["data"]['tags']).set_index("key").loc["training_completed"].value.lower()=="true"]
        
        
        if include_running == True:
            running_jobs = [r for r in runs if  r["info"]["status"] == "RUNNING" ]
            running_jobs = [r['info'] for r in running_jobs]
            comppleted_runs=comppleted_runs+running_jobs
        runs_info_df = pd.DataFrame(comppleted_runs)
        if runs_info_df.shape[0]!=0:
            runs_info_df=runs_info_df.set_index("run_id")
        
       
        
        return runs_info_df,runs




