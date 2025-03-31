import json
import os
from pathlib import Path

from mainsequence.client import TargetPortfolio, Account, AccountPortfolioScheduledRebalance, RebalanceTargetPosition

os.environ["VFB_PROJECT_PATH"] = str(Path(__file__).parent.absolute())

from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
from mainsequence import MARKETS_CONSTANTS

portfolio = PortfolioInterface.load_from_configuration("market_cap_category")
portfolio._initialize_nodes()
# res = portfolio.run(update_tree=False)
# res = portfolio.run()


target_portfolio = TargetPortfolio.get_or_none(local_time_serie__id=portfolio.portfolio_strategy_time_serie_backtest.local_metadata.id)

account = Account.get(account_name="Default MainSequence Portfolios Account")

rebalance_target_position = RebalanceTargetPosition(
    target_portfolio_id=target_portfolio.id,
    weight_notional_exposure=0.2
)
scheduled_rebalance = account.rebalance(
    target_positions=[rebalance_target_position],
    scheduled_time=None
)


