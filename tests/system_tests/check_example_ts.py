from examples.time_series.famma_franch_from_keneth import test_kenneth_french_time_serie
from examples.time_series.famma_french_from_polygon import test_fama_french_time_serie
from examples.time_series.fred_api_integration import test_fred_time_serie
from examples.time_series.no_assets_time_series import test_single_index_simulated_prices
from examples.time_series.simple_simulated_prices import test_ta_feature_simulated_crypto_prices

if __name__ == '__main__':
    test_kenneth_french_time_serie()
    test_fama_french_time_serie()
    test_fred_time_serie()
    test_single_index_simulated_prices()
    test_ta_feature_simulated_crypto_prices()
