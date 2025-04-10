from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface

from mainsequence.virtualfundbuilder.models import PortfolioConfiguration
from mainsequence.virtualfundbuilder.utils import get_vfb_logger

from pydantic import BaseModel
from mainsequence.client.models_tdag import Artifact
from mainsequence.virtualfundbuilder.resource_factory.app_factory import BaseApp, register_app

logger = get_vfb_logger()

@register_app()
class RunPortfolio(BaseApp):
    configuration_class = PortfolioConfiguration

    def __init__(self, configuration: PortfolioConfiguration):
        self.configuration = configuration

    def run(self) -> Artifact:
        portfolio = PortfolioInterface(portfolio_config_template=self.configuration.model_dump())
        res = portfolio.run()
        logger.info(f"Portfolio Run successful with results {res.head()}")


if __name__ == "__main__":
    portfolio_configuration = PortfolioInterface.load_from_configuration("market_cap_example").portfolio_config
    RunPortfolio(portfolio_configuration).run()
