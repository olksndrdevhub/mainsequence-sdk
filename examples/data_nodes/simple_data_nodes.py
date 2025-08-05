"""
These data nodes do not serve any practical purpose but only exemplify creation and best practices.



"""
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
        idx = pd.date_range((last + pd.Timedelta(days=1)) if last else today, today, freq="D", tz="UTC")
        return (
            pd.DataFrame(
                {"random_number": np.random.normal(self.mean, self.std, len(idx))},
                index=idx,
            )
            .rename_axis("time_index")
        )




