from mainsequence.client.models_tdag import Artifact
from datetime import timedelta
from typing import Union

import pandas as pd

from mainsequence.tdag.time_series import TimeSerie
from mainsequence.client import  Asset

from mainsequence.virtualfundbuilder.utils import TIMEDELTA
import numpy as np
import pandas.api.types as ptypes


class ImportValmer(TimeSerie):
    @TimeSerie._post_init_routines()
    def __init__(
            self,
            artifact_name: str,
            bucket_name: str,
            *args, **kwargs
    ):
        self.artifact_name = artifact_name
        self.bucket_name = bucket_name
        self.artifact_data = None
        super().__init__(*args, **kwargs)

    def maximum_forward_fill(self):
        return timedelta(days=1) - TIMEDELTA

    def get_explanation(self):
        explanation = (
            "### Data From Valmer\n\n"
        )
        return explanation

    def _get_artifact_data(self):
        if self.artifact_data is None:
            source_artifact = Artifact.get(bucket__name=self.bucket_name, name=self.artifact_name)
            self.artifact_data = pd.read_csv(source_artifact.content)

        return self.artifact_data

    def _get_asset_list(self) -> Union[None, list]:

        data_source = self._get_artifact_data()

        assets = []
        for figi in self.assets_source["figi"].unique():
            asset = Asset.get_or_none(figi=figi)
            if asset is None:
                try:
                    pass
                    # TODO CREATE ASSETS IN BACKEND
                    # asset = Asset.register_figi_as_asset_in_main_sequence_venue(
                    #     figi=figi,
                    #     execution_venue__symbol=CONSTANTS.MAIN_SEQUENCE_EV,
                    # )

                except Exception as e:
                    print(f"Could not register asset with figi {figi}, error {e}")
                    continue
            assets.append(asset)

        return assets

    def update(self, update_statistics: "DataUpdates"):
        data_source = self._get_artifact_data()

        # needs to have vector columns
        weights_source = None
        weights_source["time_index"] = pd.to_datetime(
            weights_source["time_index"], utc=True
        )

        # convert figis in source data
        for asset in update_statistics.asset_list:
            weights_source.loc[weights_source["figi"] == asset.figi, "unique_identifier"] = asset.unique_identifier

        weights = weights_source[["time_index", "unique_identifier", "weight"]]
        weights.rename(columns={"weight": "signal_weight"}, inplace=True)
        weights.set_index(["time_index", "unique_identifier"], inplace=True)

        weights = update_statistics.filter_df_by_latest_value(weights)
        return weights


if __name__ == "__main__":
    ts = ImportValmer(
        bucket_name="Vector de precios",
        artifact_name="Vector_20250430.csv"
    )

    ts.run(
        debug_mode=True,
        force_update=True,
    )