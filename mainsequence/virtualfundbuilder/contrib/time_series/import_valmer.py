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
            self.artifact_data = pd.read_csv(source_artifact.content, encoding='latin1')

        return self.artifact_data

    def _get_asset_list(self) -> Union[None, list]:
        source_data = self._get_artifact_data()
        assets = []
        for i, row in source_data[["Instrumento", "Isin"]].iterrows():
            instrument = row["Instrumento"]
            isin = str(row["Isin"])
            if np.isnan(row["Isin"]):
                isin = None

            asset = None
            try:
                asset = Asset.get_or_register_custom_asset_in_main_sequence_venue(ticker=instrument,
                                                                                name=instrument,
                                                                                security_type=None,
                                                                                security_type_2=None,
                                                                                security_market_sector=None,
                                                                                isin=isin,
                                                                                exchange_code=None
                                                                                )
                source_data.loc[source_data["Instrumento"] == instrument, "unique_identifier"] = asset.unique_identifier
                assets.append(asset)
            except Exception as e:
                print(f"Could not register asset with Instrumento {instrument} and Isin {isin}, error {e}")
                continue

        self.source_data = source_data
        return assets

    def update(self, update_statistics: "DataUpdates"):
        source_data = self.source_data

        assert source_data is not None, "Source data is not available"

        source_data.rename(columns={"Instrumento": "unique_identifier", "Fecha": "time_index"}, inplace=True)
        source_data['time_index'] = pd.to_datetime(source_data['time_index'], utc=True)

        # make columns lower case
        for col in source_data.columns:
            source_data.rename(columns={col: col.lower()}, inplace=True)

        source_data.set_index(["time_index", "unique_identifier"], inplace=True)
        return source_data

if __name__ == "__main__":
    ts = ImportValmer(
        bucket_name="Vector de precios",
        artifact_name="Vector_20250430.csv",
        local_kwargs_to_ignore=["bucket_name", "artifact_name"]
    )

    ts.run(
        debug_mode=True,
        force_update=True,
    )