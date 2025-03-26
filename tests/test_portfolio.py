import json
import os
from pathlib import Path

from mainsequence.client import TargetPortfolio, Account, AccountPortfolioScheduledRebalance

os.environ["VFB_PROJECT_PATH"] = str(Path(__file__).parent.absolute())

from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
from mainsequence import VAM_CONSTANTS

portfolio = PortfolioInterface.load_from_configuration("market_cap_category")
portfolio._initialize_nodes()
# res = portfolio.run(update_tree=False)
# res = portfolio.run()


target_portfolio = TargetPortfolio.get_or_none(local_time_serie__id=portfolio.portfolio_strategy_time_serie_backtest.local_metadata.id)

account = Account.get(account_name="Default MainSequence Portfolios Account")

scheduled_rebalance = account.rebalance(
    target_portfolio=target_portfolio,
    weight_notional_exposure=0.5
)


