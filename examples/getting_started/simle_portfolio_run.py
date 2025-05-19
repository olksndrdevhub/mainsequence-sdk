
#comment out for local testing out of Main Sequence Platform


from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
from mainsequence.virtualfundbuilder.contrib.time_series.external_weights import WeightsFromCSV
from mainsequence.client import SessionDataSource,Asset,CONSTANTS
#
# PortfolioInterface.list_configurations()
#
# portfolio = PortfolioInterface.load_from_configuration("market_cap")
#
#
# res = portfolio.run(local_database=True)
# res.head()





SessionDataSource.set_local_db()
ts=WeightsFromCSV(csv_file_path="/home/jose/tdag/data/weights_example.csv",
                  signal_assets_configuration=None)

ts.run(debug_mode=True,force_update=True)
a=5