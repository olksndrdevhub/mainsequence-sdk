import pytz
import pandas as pd
import datetime
import numpy as np
import dotenv
dotenv.load_dotenv('../../.env')
from mainsequence.tdag import TimeSerie
from mainsequence.client import DataUpdates

class SingleIndexSimulatedPrices(TimeSerie):
    OFFSET_START = datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)
    CPUS = 1
    GPUS = 0

    @TimeSerie._post_init_routines()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, update_statistics: DataUpdates):
        initial_price = 100.0
        mu = 0.0
        sigma = 0.01

        #because this time series is not multiindex we use max time in update statistics
        last_update = update_statistics._max_time_in_update_statistics
        sim_start = last_update + datetime.timedelta(hours=1)
        yesterday_midnight = datetime.datetime.now(pytz.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - datetime.timedelta(days=1)

        time_range = pd.date_range(start=sim_start, end=yesterday_midnight, freq='H')

        if len(time_range) == 0:
            return pd.DataFrame()

        random_returns = np.random.lognormal(mean=mu, sigma=sigma, size=len(time_range))
        simulated_prices = initial_price * np.cumprod(random_returns)

        df = pd.DataFrame({'close': simulated_prices}, index=time_range)
        df.index.name = "time_index"

        return df

def test_single_index_simulated_prices():
    ts = SingleIndexSimulatedPrices()

    # CASE 1: initial data
    print("=== Initial Simulation ===")
    ts.run(debug_mode=True, force_update=True)

# Run the test
if __name__ == "__main__":
    test_single_index_simulated_prices()
