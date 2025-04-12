
"""
This script is used to Mock the Process of integrating and execution engine into the Main Sequence

"""


## GEt
from mainsequence.client import TargetPortfolio,Account

#%% Configuration
PORTFOLIO_TICKER="portfo135B"
TARGET_ACCOUNT_ID=6



#%%

portfolio_to_mock_trading=TargetPortfolio.filter(portfolio_ticker=PORTFOLIO_TICKER)

a=5

#%% Get Account to rebalance the portfolio
account=Account.get(pk=TARGET_ACCOUNT_ID)

a=5

