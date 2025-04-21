
#comment out for local testing out of Main Sequence Platform
import dotenv

from mainsequence.virtualfundbuilder.contrib.prices.time_series import get_interpolated_prices_timeseries

dotenv.load_dotenv('../.env.dev')

from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
from mainsequence.virtualfundbuilder.agent_interface import TDAGAgent
from mainsequence.virtualfundbuilder.contrib.time_series import MarketCap
portfolio = PortfolioInterface.load_from_configuration("market_cap_example")

portfolio.run(add_portfolio_to_markets_backend=True)

# bars_ts = get_interpolated_prices_timeseries(portfolio.portfolio_build_configuration.assets_configuration)

# bars_ts.run(debug_mode=True, force_update=True)
# res.head()

