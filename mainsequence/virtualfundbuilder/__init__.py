__version__ = '0.1.0'
from dotenv import load_dotenv
from pathlib import Path
import os
import sys

def load_env():

    assert os.environ.get("VFB_PROJECT_PATH", None) is not None, "VFB_PROJECT_PATH environment variable not set"

    from mainsequence.tdag.config import Configuration
    # this step is needed to assure env variables are passed to ray cluster
    Configuration.add_env_variables_to_registry(["VFB_PROJECT_PATH"])

    sys.path.append(str(Path(os.environ.get("VFB_PROJECT_PATH")).parent))

load_env()
from mainsequence.virtualfundbuilder.utils import (
    GECKO_SYMBOL_MAPPING,
    TIMEDELTA,
    reindex_df,
    convert_to_binance_frequency,
    get_last_query_times_per_asset,
    build_rolling_regression_from_df,
    filter_assets
)

def register_default_strategies():
    # Keep this in a function to not clutter the libs namespace
    import mainsequence.virtualfundbuilder.contrib.signals
    import mainsequence.virtualfundbuilder.contrib.rebalance_strategies
register_default_strategies()
