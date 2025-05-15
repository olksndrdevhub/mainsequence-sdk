from pathlib import Path

import pandas as pd
import datetime
import dotenv

from mainsequence.virtualfundbuilder import TIMEDELTA
from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
from mainsequence.virtualfundbuilder.resource_factory.signal_factory import WeightsBase, register_signal_class

dotenv.load_dotenv('../../.env')

from mainsequence.tdag import TimeSerie, ModelList
from mainsequence.client.models_tdag import DataUpdates
from mainsequence.client.models_vam import Asset

@register_signal_class(register_in_agent=False)
class ExternalWeights(WeightsBase, TimeSerie):
    @TimeSerie._post_init_routines()
    def __init__(self, csv_path: str, *args, **kwargs):
        self.csv_path = csv_path
        super().__init__(*args, **kwargs)

    def update(self, update_statistics: DataUpdates) -> pd.DataFrame:
        # import from file path
        data = pd.read_csv(self.csv_path)
        data["time_index"] = pd.to_datetime(
            data["time_index"],
            utc=True,      # store as UTC to avoid mixed offsets/DST issues
            errors="raise"
        )

        figis = data["figi"].unique()
        assets = Asset.filter(figi__in=list(figis))

        # convert figis to assets
        for asset in assets:
            data.loc[data["figi"]==asset.figi, "unique_identifier"] = asset.unique_identifier

        data = data[["time_index", "unique_identifier", "signal_weight"]]
        if update_statistics:
            new_data_list = []
            for a in assets:
                new_data_asset = data[data["time_index"] > update_statistics[a.unique_identifier]]
                new_data_list.append(new_data_asset)
            data = pd.concat(new_data_list)

        data.set_index(["time_index", "unique_identifier"], inplace=True)
        return data

    def maximum_forward_fill(self):
        # TODO make part of configuration
        return datetime.timedelta(days=1) - TIMEDELTA


if __name__ == "__main__":
    csv_path = str(Path(__file__).parent.parent / "sample_data/portfolio_weights_mag7.csv")
    # ts = ExternalWeights(csv_path=csv_path)
    # ts.run(debug_mode=True, force_update=True)

    portfolio = PortfolioInterface.load_from_configuration("external_portfolio")
    portfolio.portfolio_build_configuration.backtesting_weights_configuration.signal_weights_configuration["csv_path"] = csv_path
    portfolio.run(add_portfolio_to_markets_backend=True)
