import pandas as pd
import datetime
import logging
import pandas_market_calendars as mcal

from mainsequence.virtualfundbuilder.enums import ResourceType
from mainsequence.virtualfundbuilder.resource_factory.base_factory import BaseFactory, BaseResource, insert_in_registry

logger = logging.getLogger("virtualfundbuilder")

class RebalanceStrategyBase(BaseResource):
    TYPE = ResourceType.REBALANCE_STRATEGY

    def __init__(self,
                 calendar: str='24/7',
                 *args, **kwargs
    ):
        """
        Args:
            calendar (str): Trading calendar. The string should must be valid calendar from the pandas_market_calendars (like '24/7' or 'NYSE')
        """
        self._calendar = calendar

    @property
    def calendar(self):
        """ Workaround due to error when pickleing the calendar """
        return mcal.get_calendar(self._calendar)

    def get_explanation(self):
        info = f"""
        <p>{self.__class__.__name__}: Rebalance strategy class.</p>
        """
        return info

    def calculate_rebalance_dates(
            self,
            start: datetime.datetime,
            end: datetime.datetime,
            calendar,
            rebalance_frequency_strategy: str
    ) -> pd.DatetimeIndex:
        """
        Determines the dates on which portfolio rebalancing should be executed based on the specified rebalancing strategy.
        This calculation takes into account the start time of the rebalancing window and the execution frequency.

        Args:
            start (pd.DataFrame): A datetime containing the start time

        Returns:
            pd.DatetimeIndex: A DatetimeIndex containing all the dates when rebalancing should occur.
        """
        # to account for the time during the day at which the execution starts
        if end is None:
            raise NotImplementedError("end_date cannot be None")

        if rebalance_frequency_strategy == "daily":
            early = calendar.schedule(start_date=start.date(), end_date=end.date())
            rebalance_dates = early.set_index("market_open")
            rebalance_dates = rebalance_dates.index
        elif rebalance_frequency_strategy == "EOQ":
            # carefull to use dates from the same calendar
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Strategy {rebalance_frequency_strategy} not implemented")

        return rebalance_dates

REBALANCE_CLASS_REGISTRY = REBALANCE_CLASS_REGISTRY if 'REBALANCE_CLASS_REGISTRY' in globals() else {}
def register_rebalance_class(name=None, register_in_agent=True):
    """
    Decorator to register a model class in the factory.
    If `name` is not provided, the class's name is used as the key.
    """
    def decorator(cls):
        return insert_in_registry(REBALANCE_CLASS_REGISTRY, cls, register_in_agent, name)
    return decorator

class RebalanceFactory(BaseFactory):

    @staticmethod
    def get_rebalance_strategy(rebalance_strategy_name: str):
        if rebalance_strategy_name not in REBALANCE_CLASS_REGISTRY:
            RebalanceFactory.get_rebalance_strategies()
        try:
            return REBALANCE_CLASS_REGISTRY[rebalance_strategy_name]
        except KeyError:
            logger.exception(f"{rebalance_strategy_name} is not registered in this project")

    @staticmethod
    def get_rebalance_strategies():
        import mainsequence.virtualfundbuilder.contrib.rebalance_strategies # get default strategies
        try:
            RebalanceFactory.import_module("rebalance_strategies")
        except FileNotFoundError:
            logger.info("rebalance_strategies folder no present no strategies to import")
        return REBALANCE_CLASS_REGISTRY





