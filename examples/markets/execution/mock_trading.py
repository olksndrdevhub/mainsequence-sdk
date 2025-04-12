
"""
This script is used to Mock the Process of integrating and execution engine into the Main Sequence

"""


## GEt
from mainsequence.client import TargetPortfolio,Account,RebalanceTargetPosition,RebalanceTargetPosition

#%% Configuration
PORTFOLIO_TICKER="portfo135B"
TARGET_ACCOUNT_ID=6



#%%

portfolio_to_mock_trading=TargetPortfolio.get(portfolio_ticker=PORTFOLIO_TICKER)

a=5

#%% Get Account to rebalance the portfolio
account=Account.get(pk=TARGET_ACCOUNT_ID)

print("Account Target Positions")
print(account.get_latest_positions_as_df())

rebalance_target_position = RebalanceTargetPosition(
    target_portfolio_id=portfolio_to_mock_trading.id,
    weight_notional_exposure=0.2
)
scheduled_rebalance = account.rebalance(
    target_positions=[rebalance_target_position],
    scheduled_time=None
)


a=5

