import collections
import operator

import pandas as pd
import numpy  as np

class RollingMinMax:
    def __init__(self, initial_window:list,is_min:bool ):
        self.window_size = len(initial_window)
        self.window = collections.deque()
        self.counter = 0
        self.is_min=is_min
        for x in initial_window:
            self.append_value_to_queue(x,increase_window=False)

    @property
    def value(self):
        return self.window[0][0]

    def append_value_to_queue(self, value, increase_window=True):
        comp=operator.ge if self.is_min == True else operator.le
        if increase_window == True:
            self.window_size = self.window_size + 1

        while self.window and comp(self.window[-1][0], value):
            self.window.pop()

        self.window.append((value, self.counter))

        self.counter = self.counter + 1

    def update_window(self):
        self.window_size = self.window_size - 1
        self.counter = self.counter - 1
        while self.window[0][1] <= self.counter - self.window_size:
            self.window.popleft()


class RollingDistances:
    def __init(self,rolling_window:int,):
        """

        Parameters
        ----------
        rolling_window (int): The size of the rolling window. Default is 3.


        """
        self.rolling_window=rolling_window

    def dtw(self,df:pd.DataFrame):
        """
           Calculate the rolling dynamic time warp (DTW) distances between all columns of a DataFrame.

           Parameters:
               df (pandas.DataFrame): The input DataFrame containing the columns.


           Returns:
               dict: A dictionary where the keys are tuples representing column pairs,
                     and the values are lists of DTW distances corresponding to the rolling windows.
           """
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        result = {}

        columns = df.columns
        num_columns = len(columns)

        data = df.values

        for i in range(num_columns):
            for j in range(i + 1, num_columns):
                col1 = data[:, i]
                col2 = data[:, j]

                distances = np.zeros(len(col1) - self.rolling_window + 1)
                for k in range( self.rolling_window):
                    subseq1 = col1[k:len(col1) -  self.rolling_window + k + 1]
                    subseq2 = col2[k:len(col2) -  self.rolling_window + k + 1]

                    _, path = fastdtw(subseq1, subseq2, dist=euclidean)
                    distance = sum([subseq1[idx] for idx, _ in path]) / len(path)
                    distances += distance

                result[(columns[i], columns[j])] = distances.tolist()

        return result



