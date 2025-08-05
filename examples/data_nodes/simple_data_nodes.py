"""
These data nodes do not serve any practical purpose but only exemplify creation and best practices.



"""
from typing import Dict, Union

import pandas as pd

from mainsequence.client import UpdateStatistics
from mainsequence.tdag.data_nodes import DataNode
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



def build_test_time_series():
    daily_node = DailyRandomNumber(mean=0.0, std=1.0)


    daily_node.run(debug_mode=True, force_update=True)


