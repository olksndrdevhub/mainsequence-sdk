import os
from pathlib import Path
os.environ["VFB_PROJECT_PATH"] = str(Path(__file__).parent.absolute())

from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface

portfolio = PortfolioInterface.load_from_configuration("market_cap_top10_tech")
res = portfolio.run()