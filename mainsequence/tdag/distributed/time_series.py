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

class RemoteTaskTrainTimeSerie(TimeSerie):
    TRAIN_NAMESPACE = "TIMESERIE_TRAIN"

    def __init__(self,training_project_name:Union[str,None],
                 retrain_target_time_delta_seconds: Union[float, None],
                 *args, **kwargs):
        """

        Parameters
        ----------
        training_project_name :
        retrain_target_time_delta_seconds : (int) this parameter allows for protection in setting ahigher training
        frequency for the update of the ts but only train on succesfull model iterations.
        from example ts update can be veryday but actuall training happens only on monthly intervals between succesfull
        iterations


        args :
        kwargs :
        """


        build_meta_data = {"task_config":{"num_cpus":1}} if "build_meta_data" not in kwargs else kwargs["build_meta_data"]
        kwargs["build_meta_data"]=build_meta_data
        super().__init__(*args, **kwargs)
        project_name = training_project_name
        if training_project_name is  None:
            project_name=f"default_{self.__class__.__name__}"
        self.training_project_name=project_name
        self.tracking_api = MLflowTrackingRestApi()
        self.retrain_target_time_delta_seconds=retrain_target_time_delta_seconds

    @property
    def human_readable(self):
        return f"{self.training_project_name}"

    def _train(self, run_name, custom_run_config: dict,        )->str:
        """
        Main Training function should implement a ray remote task
        Parameters
        ----------
        run_name :

        Returns
        -------

        """
        raise NotImplementedError

    def _train_debug(self, run_name: str, custom_run_config: dict) -> str:
        run_id = self._train(run_name=run_name,custom_run_config=custom_run_config)
        return run_id

    def _get_custom_run_config(self, update_tree_kwargs: dict):
        custom_run_config = {}
        if self.hash_id in update_tree_kwargs.keys():
            custom_run_config = update_tree_kwargs[self.hash_id]
        return custom_run_config

    def update_series_from_source(self, latest_value, **class_arguments):
        """

        Parameters
        ----------
        latest_value :
        class_arguments :

        Returns
        -------

        """
        custom_run_config = self._get_custom_run_config(update_tree_kwargs=class_arguments["update_tree_kwargs"])
        
        last_run = None

        latest_value = datetime.datetime(2021, 1, 1).replace(tzinfo=pytz.utc) if latest_value is None else latest_value
        train_time = datetime.datetime.now(pytz.utc)
        # always retrain on retrain delta
        temp_df, last_run = self.last_trained_model
        
        #check if jobs are running
        running_jobs=self.running_jobs
        running_jobs=pd.DataFrame()
        if running_jobs.shape[0] !=0:
            self.logger.info("Jobs are running no update needeed")
            return pd.DataFrame()
        self.logger.exception("Change the running jbos to !=0")
        
        force_retrain = False if "force_train" not in custom_run_config else custom_run_config['force_train']
        last_timestamp_on_train = self.last_trained_model_last_timestamp_on_train
        last_timestamp_on_train=last_timestamp_on_train if last_timestamp_on_train is not None else latest_value
        if last_run is None:
            force_retrain= True
        

        retrain_by_delta = False if self.retrain_target_time_delta_seconds is None else (
                                                                                                    train_time - last_timestamp_on_train).total_seconds() > self.retrain_target_time_delta_seconds

        if force_retrain == False and retrain_by_delta == False:
            self.logger.info(f"Not Training Model force_retrain {force_retrain}  retrain_by_delta {retrain_by_delta}")
            return pd.DataFrame()
        self.logger.warning("Training model ...")
        # train
        run_name = str(train_time.timestamp())
        if class_arguments["update_tree_kwargs"]["DEBUG"] == True:
            self.logger.warning("Training model in debug mode ...")
            run_name = self._train_debug(run_name=run_name,
                                         custom_run_config=custom_run_config,
                                         )
        else:

          

            self._train(run_name=run_name,custom_run_config=custom_run_config                )

        temp_df, last_run = self.last_trained_model
        temp_df = temp_df[temp_df.index > latest_value]

        return temp_df

    @property
    def production_run(self):
        """
        Gets model with tag "production":True if more than one model has production flag then raises error
        Returns
        -------

        """
        all_finished_runs, runs = self.tracking_api.get_all_finished_runs_df(experiment_name=self.training_project_name)
       
        production_run=[r for r in runs if len([t for t in r['data']['tags'] if t['key']=="production_model" ])>0 ]
       
        if len(production_run)!=1:
            raise Exception(f"The number of prediction models is {len(production_run)}")
        return production_run[0]
    
    @property
    def running_jobs(self):
        all_finished_runs, runs = self.tracking_api.get_all_finished_runs_df(experiment_name=self.training_project_name,
                                                                             include_running=True)
        if all_finished_runs.shape[0]==0:
            return  all_finished_runs
        all_finished_runs=all_finished_runs[all_finished_runs["status"]=="RUNNING"]
        return all_finished_runs
    @property
    def last_trained_model(self):

        all_finished_runs, runs = self.tracking_api.get_all_finished_runs_df(experiment_name=self.training_project_name)
        if all_finished_runs.shape[0] == 0:
            return all_finished_runs, None
        last_run = all_finished_runs.reset_index().sort_values("end_time", ascending=True).iloc[-1]
        temp_df = pd.DataFrame(last_run.to_dict(), index=[0]
                               ).set_index("end_time")
        last_run = [r for r in runs if r["info"]["run_id"] == temp_df.iloc[0]["run_id"]][0]
        temp_df.index = [datetime.datetime.utcfromtimestamp(temp_df.index[0] * 1e-3).replace(tzinfo=pytz.utc)]

        return temp_df, last_run
    @property
    def last_trained_model_last_timestamp_on_train(self):
        temp_df, last_run=self.last_trained_model
        last_timestamp_on_train=None

        if last_run is not None:
            last_timestamp_on_train=[c['value'] for c in last_run['data']['tags'] if c["key"]=='last_timestamp_on_train'][0]
            if last_timestamp_on_train and last_timestamp_on_train != "None":
                last_timestamp_on_train=pd.to_datetime(last_timestamp_on_train)
        return last_timestamp_on_train


