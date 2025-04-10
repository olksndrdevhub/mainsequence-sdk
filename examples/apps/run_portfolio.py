from mainsequence.virtualfundbuilder.models import PortfolioConfiguration
from mainsequence.virtualfundbuilder.utils import get_vfb_logger

from pydantic import BaseModel
from mainsequence.client.models_tdag import Artifact
from mainsequence.virtualfundbuilder.resource_factory.app_factory import BaseApp, register_app

logger = get_vfb_logger()

@register_app(register_in_agent=False)
class RunPortfolio(BaseApp):
    configuration_class = PortfolioConfiguration

    def __init__(self, configuration: PortfolioConfiguration):
        self.configuration = configuration
        pass

    def run(self) -> Artifact:
        pass


class NamedPortfolioConfiguration(BaseModel):
    portfolio_name: str = "market_cap_example"
    update_tree: bool = True

@register_app()
class RunNamedPortfolio(BaseApp):
    configuration_class = NamedPortfolioConfiguration

    def __init__(self, configuration: NamedPortfolioConfiguration):
        logger.info(f"Run Named Timeseries Configuration {configuration}")
        self.configuration = configuration

    def run(self) -> None:
        from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
        portfolio = PortfolioInterface.load_from_configuration(self.configuration.portfolio_name)
        res = portfolio.run(update_tree=self.configuration.update_tree)
        logger.info(f"Portfolio Run successful with results {res.head()}")

if __name__ == "__main__":
    configuration = NamedPortfolioConfiguration()
    RunNamedPortfolio(configuration).run()
