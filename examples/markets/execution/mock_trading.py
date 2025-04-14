
"""
This script is used to Mock the Process of integrating and execution engine into the Main Sequence

"""


## GEt
from mainsequence.client import TargetPortfolio,Account,RebalanceTargetPosition,RebalanceTargetPosition
from mainsequence.client import AssetOnlyPortfolio
#%% Configuration
PORTFOLIO_TICKER="portfo135B"
TARGET_ACCOUNT_ID=6



#%%

portfolio_to_mock_trading=TargetPortfolio.get(portfolio_ticker=PORTFOLIO_TICKER)





#%% Get Account to rebalance the portfolio
account=Account.get(pk=TARGET_ACCOUNT_ID,timeout=10000000)
asset_only_portfolio=AssetOnlyPortfolio.get(tracking_asset__id=account.cash_asset.id)
print("Account Latest Holdings")
print(account.latest_holdings)
print("Account Target Positions")
print(account.account_target_portfolio.latest_positions)

make_rebalance=True
for p in account.account_target_portfolio.latest_positions.positions:
    if p.target_portfolio==portfolio_to_mock_trading.id:
        make_rebalance=False
        break

if make_rebalance:
    rebalance_target_position_port = RebalanceTargetPosition(
        target_portfolio_id=portfolio_to_mock_trading.id,
        weight_notional_exposure=0.2
    )
    rebalance_target_position_cash = RebalanceTargetPosition(
        target_portfolio_id=asset_only_portfolio.id,
        weight_notional_exposure=0.8
    )
    scheduled_rebalance = account.rebalance(
        target_positions=[rebalance_target_position_port,rebalance_target_position_cash],
        scheduled_time=None,
    )

# A this point we should have a new target portfolio set and a scheduled rebalance that should have happened

#we are going to mock an execution engine that is always forcing to match  the tracking error of our portfolio with
#first lets snapshot the account to guaranteed las valuation
account.snapshot_account(timeout=1000000)

#second lets get the tracking error detail
fund_summary,account_tracking_error=account.get_tracking_error_details(timeout=1000000)
print(fund_summary) #give us the state of the fund





