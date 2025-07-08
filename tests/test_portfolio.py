
#comment out for local testing out of Main Sequence Platform
import dotenv

from mainsequence.client import SessionDataSource
from mainsequence.virtualfundbuilder.contrib.prices.time_series import get_interpolated_prices_timeseries

dotenv.load_dotenv('../.env.dev')

from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
portfolio = PortfolioInterface.load_from_configuration("market_cap")

# SessionDataSource.set_local_db()
res = portfolio.run(update_tree=False)
print(res)
# bars_ts = get_interpolated_prices_timeseries(portfolio.portfolio_build_configuration.assets_configuration)

# bars_ts.run(debug_mode=True, force_update=True)
# res.head()

