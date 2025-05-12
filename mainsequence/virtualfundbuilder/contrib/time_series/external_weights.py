
from mainsequence.tdag.time_series import WrapperTimeSerie
from mainsequence.client import CONSTANTS, Asset, AssetTranslationTable,  DoesNotExist

from datetime import datetime, timedelta, tzinfo
from typing import Optional, List, Union

import pandas as pd

from mainsequence.tdag.time_series import TimeSerie
from mainsequence.client import  Asset, AssetCategory

from mainsequence.virtualfundbuilder.resource_factory.signal_factory import WeightsBase, register_signal_class
from mainsequence.virtualfundbuilder.utils import TIMEDELTA
import numpy as np
import pandas.api.types as ptypes





@register_signal_class(register_in_agent=True)
class WeightsFromCSV(WeightsBase, TimeSerie):
    @TimeSerie._post_init_routines()
    def __init__(self,
                 csv_file_path: str,

                 *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.csv_file_path = csv_file_path


    def maximum_forward_fill(self):
        return timedelta(days=1) - TIMEDELTA

    def get_explanation(self):


        explanation = (
            "### External Weights Source\n\n"
            f"This strategy represents weights from an external source:\n\n"

        )

        return explanation

    def _read_weights(self):
        weights_source = pd.read_csv(self.csv_file_path)

        # exactly these columns, in order
        expected_cols = [
            "time_index",
            "signal_weight",
            "ticker",
            "exchange_code",
            "security_type",
            "security_type_2",
            "market_sector",
        ]
        if list(weights_source.columns) != expected_cols:
            raise ValueError(
                f"Invalid CSV format: expected columns {expected_cols!r} "
                f"(in that order), but got {list(weights_source.columns)!r}"
            )

        # time_index → integer Unix timestamp
        if not ptypes.is_integer_dtype(weights_source["time_index"]):
            raise ValueError(
                f"'time_index' must be integer Unix timestamps, but dtype is "
                f"{weights_source['time_index'].dtype}"
            )
        weights_source["time_index"] = pd.to_datetime(
            weights_source["time_index"], unit="s", utc=True
        )

        # signal_weight → numeric
        if not ptypes.is_numeric_dtype(weights_source["signal_weight"]):
            raise ValueError(
                f"'signal_weight' must be numeric, but dtype is "
                f"{weights_source['signal_weight'].dtype}"
            )

        # metadata columns → non-null strings
        for col in [
            "ticker",
            "exchange_code",
            "security_type",
            "security_type_2",
            "market_sector",
        ]:

            if not ptypes.is_object_dtype(weights_source[col]):
                raise ValueError(
                    f"'{col}' must be string/object dtype, but is "
                    f"{weights_source[col].dtype}"
                )

        return weights_source

    def _get_asset_list(self) -> Union[None, list]:

        weights_df=self._read_weights()
        weights_df['map_search'] = (
                weights_df['ticker']
                + weights_df['exchange_code']
                + weights_df['security_type']
                + weights_df['security_type_2']
                + weights_df['market_sector']
        )
        self.weights_df=weights_df
        asset_list: List[Asset] = []

        for _, row in weights_df.iterrows():
            results = Asset.filter(
                ticker=row['ticker'],
                execution_venue=CONSTANTS.MAIN_SEQUENCE_EV,
                exchange_code=row['exchange_code'],
                security_type=row['security_type'],
                security_subtype=row['security_type_2'],
                market_sector=row['market_sector'],
            )
            if len(results) == 0:
                raise Exception("Asset Not found {}".format(row['ticker']))
            # 4) tag each asset with the map_search key
            for asset in results:
                setattr(asset, 'map_search', row['map_search'])
                asset_list.append(asset)

        return asset_list

    def update(self, update_statistics: "DataUpdates"):
        """
        Args:
            latest_value (Union[datetime, None]): The timestamp of the most recent data point.

        Returns:
            DataFrame: A DataFrame containing updated signal weights, indexed by time and asset symbol.
        """
        asset_list = update_statistics.asset_list

        external_weights=self._read_weights()



        return signal_weights