"""
These data nodes do not serve any practical purpose but only exemplify creation and best practices.



"""
from typing import Dict, Union

import pandas as pd

from mainsequence.client import UpdateStatistics
from mainsequence.tdag.data_nodes import DataNode, APIDataNode
import mainsequence.client as ms_client
import numpy as np


class DailyRandomNumber(DataNode):

    def __init__(self,mean:float,std:float, *args, **kwargs):
        """
        :param mean:  the mean of the probability  distribution
        :param std: the std of the probability  distribution
        :param kwargs:
        """
        self.mean=mean
        self.std=std
        super().__init__(*args, **kwargs)

    def get_table_metadata(self)->ms_client.TableMetaData:
        TS_ID = f"example_random_number_{self.mean}_{self.std}"
        meta = ms_client.TableMetaData( identifier=TS_ID,
                                        description="Example Data Node")

        return meta

    def update(self) -> pd.DataFrame:
        """Draw daily samples from N(mean, std) since last run (UTC days)."""
        today = pd.Timestamp.now("UTC").normalize()
        last = self.update_statistics.max_time_index_value
        if last is not None and last >= today:
            return pd.DataFrame()
        return pd.DataFrame(
            {"random_number": [np.random.normal(self.mean, self.std)]},
            index=pd.DatetimeIndex([today], name="time_index", tz="UTC"),
        )

    def dependencies(self) -> Dict[str, Union["DataNode", "APIDataNode"]]:
        return {}


class DailyRandomAddition(DataNode):
    def __init__(self,mean:float,std:float, *args, **kwargs):
        self.mean=mean
        self.std=std
        self.daily_random_number_data_node=DailyRandomNumber(mean=0.0,std=std,*args, **kwargs)
        super().__init__(*args, **kwargs)
    def dependencies(self):
        return {"number_generator":self.daily_random_number_data_node}
    def update(self) -> pd.DataFrame:
        """Draw daily samples from N(mean, std) since last run (UTC days)."""
        today = pd.Timestamp.now("UTC").normalize()
        last = self.update_statistics.max_time_index_value
        if last is not None and last >= today:
            return pd.DataFrame()
        random_number=np.random.normal(self.mean, self.std)
        dependency_noise=self.daily_random_number_data_node.\
                get_df_between_dates(start_date=today, great_or_equal=True).iloc[0]["random_number"]
        self.logger.info(f"random_number={random_number} dependency_noise={dependency_noise}")

        return pd.DataFrame(
            {"random_number": [random_number+dependency_noise]},
            index=pd.DatetimeIndex([today], name="time_index", tz="UTC"),
        )

class DailyRandomAdditionAPI(DataNode):
    def __init__(self,mean:float,std:float,
                 dependency_identifier:int,
                 *args, **kwargs):
        self.mean=mean
        self.std=std

        self.daily_random_number_data_node=APIDataNode.build_from_identifier(identifier=dependency_identifier)
        super().__init__(*args, **kwargs)
    def dependencies(self):
        return {"number_generator":self.daily_random_number_data_node}
    def update(self) -> pd.DataFrame:
        """Draw daily samples from N(mean, std) since last run (UTC days)."""
        today = pd.Timestamp.now("UTC").normalize()
        last = self.update_statistics.max_time_index_value
        if last is not None and last >= today:
            return pd.DataFrame()
        random_number=np.random.normal(self.mean, self.std)
        dependency_noise=self.daily_random_number_data_node.\
                get_df_between_dates(start_date=today, great_or_equal=True).iloc[0]["random_number"]
        self.logger.info(f"random_number={random_number} dependency_noise={dependency_noise}")

        return pd.DataFrame(
            {"random_number": [random_number+dependency_noise]},
            index=pd.DatetimeIndex([today], name="time_index", tz="UTC"),
        )


def build_test_time_series():


    daily_node = DailyRandomNumber(mean=0.0, std=1.0)
    daily_node.run(debug_mode=True, force_update=True)

    daily_node = DailyRandomAddition(mean=0.0, std=1.0)
    daily_node.run(debug_mode=True, force_update=True)


    daily_node = DailyRandomAdditionAPI(mean=0.0, std=1.0,
                                        dependency_identifier=f"example_random_number_{0.0}_{1.0}"
                                        )
    daily_node.run(debug_mode=True, force_update=True)

