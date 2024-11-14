import datetime

import concurrent.futures
import pytz
from mainsequence.vam_client import (
                         Account, AccountCoolDown, sync_account_funds,

                        )
import time
from logging import Logger


class AccountLooper:


    def __init__(self,logger:Logger,account_sync_wait_second=30,threadpool_workers=3):
        self.logger = logger
        self.account_pool = concurrent.futures.ThreadPoolExecutor(max_workers=threadpool_workers)
        self.account_sync_wait_second=account_sync_wait_second

    def execute_account_update_loop(self):

        loop_starts = datetime.datetime.now(pytz.utc)

        try:

            accounts, _ = Account.filter(account_is_live=True)
            for account in accounts:

                self.account_pool.submit(self.update_account_loop, account=account,logger=self.logger)

        except Exception as e:
            self.logger.exception("Error on account loop")
        loop_time_cost = (datetime.datetime.now(pytz.utc) - loop_starts).total_seconds()
        delay = max(0, self.account_sync_wait_second - loop_time_cost)
        time.sleep(delay)
        self.logger.debug("Loop completed")
        self.execute_account_update_loop()

    @staticmethod
    def update_account_loop(account: Account,logger:Logger):
        """
        This methoods runs one account update
        (1) sync funds
        (2) Very risk parameters for an unwind
        Parameters
        ----------
        account :

        Returns
        -------

        """
        try:

            account, risk_factors = sync_account_funds(account=account)
        except Exception as e:
            logger.exception(e)

        # chek breaks
        if account.is_account_in_cool_down == False:

            cool_down_config = account.execution_configuration.cooldown_configuration
            if hasattr(risk_factors, "total_unrealized_profit"):
                if risk_factors.total_unrealized_profit / risk_factors.account_balance < -1 * abs(
                        cool_down_config['unrealized_loss_percent_stop_loss']):
                    return None
                    # create cooldown
                    new_cool_down = AccountCoolDown.create(related_account__id=account.id,
                                                           cool_down_ends=datetime.datetime.now(pytz.utc) + \
                                                                          datetime.timedelta(minutes=cool_down_config[
                                                                              'cool_down_period_minutes'])
                                                           )
                    # unwind account
                    try:
                        AccountExecutor.unwind_all_holdings_in_account_task(account=account)

                    except Exception as e:
                        new_cool_down.delete()
                        raise e