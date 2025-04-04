import logging
from .models import *
logger = get_vfb_logger()

class AssetUniverseParser:
    
    @staticmethod
    def parse_asset_list(asset_list):
        """
        Adjusted version that expects a top-level structure:

            'asset_list': [
                {
                    'asset_type': 'cash_equity',
                    'execution_venue__symbol': 'alpaca',
                    'symbol': 'AAPL'
                },
                ...
            ]
        """
        # Turn the asset_list into a DataFrame for easier group-by logic
        df_all = pd.DataFrame(asset_list)
        if df_all.empty:
            logger.warning("asset_list is empty after DataFrame creation.")
            return asset_list

        # We will build a final list of all valid assets
        final_assets = []

        # --- 1) Process includes/excludes by grouping on execution venue
        for exec_venue_symbol, df_venue in df_all.groupby("execution_venue__symbol"):
            # df_venue => all rows for that execution venue
            # Convert df_venue to a DataFrame we can manipulate row-by-row
            df_venue = df_venue.copy()

            # We collect new rows to add and a set of symbols to exclude
            assets_to_add = []
            assets_to_exclude = set()

            # Group by asset_type so we can parse special group types
            for asset_type, asset_group in df_venue.groupby("asset_type"):

                # convenience function to expand any special SP500 groups
                def parse_group(symbol):
                    if symbol == "SP500":
                        return SP500_MAP
                    elif symbol == "SP500_LOW_ESG":
                        return SP500_LOW_ESG
                    elif symbol == "SP500_HIGH_ESG":
                        return SP500_HIGH_ESG
                    else:
                        raise NotImplementedError(f"Unknown group_to_exclude '{symbol}'")

                if asset_type.endswith("__group__exclude"):
                    # e.g. 'cash_equity__group__exclude'
                    for symbol in asset_group["symbol"].tolist():
                        group_to_exclude = parse_group(symbol)
                        # Exclude each symbol in the group
                        group_excluded_symbols = [a["symbol"] for a in group_to_exclude]
                        assets_to_exclude.update(group_excluded_symbols)
                        # Exclude the group keyword itself
                        assets_to_exclude.add(symbol)

                elif asset_type.endswith("__exclude"):
                    # e.g. 'cash_equity__exclude'
                    for symbol in asset_group["symbol"].tolist():
                        assets_to_exclude.add(symbol)

                elif asset_type.endswith("__group"):
                    # e.g. 'cash_equity__group'
                    for symbol in asset_group["symbol"].tolist():
                        group_to_add = parse_group(symbol)
                        # We extend assets_to_add with the entire group
                        assets_to_add.extend(group_to_add)
                        # Also exclude the group keyword itself
                        assets_to_exclude.add(symbol)

                # else: normal asset_type, nothing special to do here

            # Now build a combined set of original + newly added (minus excluded)
            old_assets = df_venue.to_dict("records")
            old_assets.extend(assets_to_add)
            # Filter out anything whose symbol is in assets_to_exclude
            filtered_assets = [a for a in old_assets if a["symbol"] not in assets_to_exclude]

            logger.info(
                f"For venue='{exec_venue_symbol}': added={len(assets_to_add)} excluded={len(assets_to_exclude)}"
            )

            # Keep track of the filtered assets (still in dictionary form).
            final_assets.extend(filtered_assets)

        # --- 2) Drop excluded/untracked assets by checking with your cached_asset_filter
        # Now final_assets is across all execution venues, so we'll group again
        df_filtered = pd.DataFrame(final_assets)
        if df_filtered.empty:
            logger.warning("No assets remain after includes/excludes.")
            return []

        # Another pass grouping by (execution_venue__symbol, asset_type)
        # so we can check which exist in your internal system
        to_drop_symbols = set()
        for (exec_venue_symbol, asset_type), df_grp in df_filtered.groupby(
                ["execution_venue__symbol", "asset_type"]
        ):
            tmp_assets = cached_asset_filter(
                asset_type, exec_venue_symbol, tuple(df_grp["symbol"].tolist())
            )
            existing_symbols = {a.symbol for a in tmp_assets}
            # Everything not in existing_symbols gets dropped
            symbols_to_drop = set(df_grp["symbol"].tolist()) - existing_symbols
            to_drop_symbols.update(symbols_to_drop)

        logger.info(f"Dropping unsupported assets: {to_drop_symbols}")

        # Filter them out
        df_final = df_filtered[~df_filtered["symbol"].isin(to_drop_symbols)]

        # Replace the original asset_list with our final list
        asset_list = df_final.to_dict("records")

        # If, after all filtering, no assets remain, log a warning
        if df_final.empty:
            logger.warning("There are no assets in the final universe.")

        return asset_list


def replace_none_with_python_none(config):
    """
    Recursively replace all string 'None' with Python None in the given dictionary
    and log the path where replacements occur.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: Updated dictionary with 'None' replaced by Python None.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    def recursive_replace(d, path="root"):
        if isinstance(d, dict):
            for key, value in d.items():
                current_path = f"{path}.{key}"
                if isinstance(value, dict):
                    recursive_replace(value, current_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        recursive_replace(item, f"{current_path}[{i}]")
                elif value == 'None':
                    d[key] = None
                    logger.info(f"Replaced 'None' in configuration with None at {current_path}")
        elif isinstance(d, list):
            for i, item in enumerate(d):
                recursive_replace(item, f"{path}[{i}]")

    recursive_replace(config)
    return config

def configuration_sanitizer(configuration: dict, auto_complete=False) -> PortfolioConfiguration:
    """
    Verifies that a configuration has all the required attributes.

    If `auto_complete` is True, missing parts of the configuration will be auto-completed when possible.

    Args:
        configuration (dict): The configuration dictionary to sanitize.
        auto_complete (bool, optional): Whether to auto-complete missing parts. Defaults to False.

    Returns:
        PortfolioConfiguration: The sanitized portfolio configuration.
    """

    DEFAULT_TDAG_UPDATE_CONFIGURATION = {
        'base_dependency_tree_update_details': {
            'update_schedule': "*/1 * * * *",
            'distributed_num_cpus': 1,
            'execution_timeout_seconds': 50,
        },
        'base_classes_to_exclude': [],
        'custom_update_details_per_class': {},
    }

    DEFAULT_PRICES_CONFIGURATION = {
        'bar_frequency_id': "1days",  # "1m",
        'upsample_frequency_id': "1days",  # "15m",
        'intraday_bar_interpolation_rule': "ffill",
    }
    configuration = copy.deepcopy(configuration)
    # Review portfolio_build_configuration
    configuration = replace_none_with_python_none(configuration)
    portfolio_build_config = configuration["portfolio_build_configuration"]
    for key in ["assets_configuration", "backtesting_weights_configuration", "execution_configuration"]:
        if key not in portfolio_build_config:
            raise KeyError(f"Missing required key {key}")

    # Prices configuration review
    if portfolio_build_config["assets_configuration"] is not None:
        if "prices_configuration" not in portfolio_build_config["assets_configuration"]:
            if not auto_complete:
                raise Exception("Missing prices configuration in portfolio_build_config['assets_configuration']")
            portfolio_build_config["assets_configuration"]["prices_configuration"] = DEFAULT_PRICES_CONFIGURATION

    if "rebalance_strategy_configuration" not in portfolio_build_config["backtesting_weights_configuration"]:
        if not auto_complete:
            raise Exception(
                "Missing 'rebalance_strategy_configuration' in 'backtesting_weights_configuration'"
            )
        portfolio_build_config["backtesting_weights_configuration"]["rebalance_strategy_configuration"] = {}
        logger.warning("rebalance_strategy_configuration missing - using default empty dict")

    if "calendar" not in portfolio_build_config["backtesting_weights_configuration"]["rebalance_strategy_configuration"] or not portfolio_build_config["backtesting_weights_configuration"]["rebalance_strategy_configuration"]["calendar"]:
        if not auto_complete:
            raise Exception(
                "Missing 'calendar' in 'rebalance_strategy_configuration'"
            )
        portfolio_build_config["backtesting_weights_configuration"]["rebalance_strategy_configuration"]["calendar"] = "24/7"
        logger.warning("calendar missing in rebalance strategy configuration - using default '24/7'")

    if "signal_weights_configuration" not in portfolio_build_config["backtesting_weights_configuration"]:
        if not auto_complete:
            raise Exception(
                "Missing 'signal_weights_configuration' in 'backtesting_weights_configuration'"
            )
        portfolio_build_config["backtesting_weights_configuration"]["signal_weights_configuration"] = {}
        logger.warning("signal_weights_configuration missing - using default empty dict")

    configuration["portfolio_build_configuration"] = portfolio_build_config

    # Review tdag configuration
    if "portfolio_tdag_update_configuration" not in configuration:
        if not auto_complete:
            raise Exception("Missing 'portfolio_tdag_update_configuration'")
        configuration["portfolio_tdag_update_configuration"] = DEFAULT_TDAG_UPDATE_CONFIGURATION

    # Review portfolio_vam_configuration
    portfolio_vam_config = configuration['portfolio_vam_configuration']

    if "builds_from_target_positions" not in portfolio_vam_config:
        if not auto_complete:
            raise Exception("Missing 'builds_from_target_positions' in portfolio_vam_config")
        portfolio_vam_config["builds_from_target_positions"] = True

    if "tracking_funds_expected_exposure_from_latest_holdings" not in portfolio_vam_config:
        if not auto_complete:
            raise Exception(
                "Missing 'tracking_funds_expected_exposure_from_latest_holdings' in portfolio_vam_config"
            )
        portfolio_vam_config["tracking_funds_expected_exposure_from_latest_holdings"] = False

    if "follow_account_rebalance" not in portfolio_vam_config:
        if not auto_complete:
            raise Exception("Missing 'follow_account_rebalance' in portfolio_vam_config")
        portfolio_vam_config["follow_account_rebalance"] = True

    configuration['portfolio_vam_configuration'] = portfolio_vam_config


    # Add asset groups or remove excluded assets
    asset_universe_config = portfolio_build_config["assets_configuration"]["asset_universe"]
    portfolio_build_config["assets_configuration"]["asset_universe"] = AssetUniverse(**asset_universe_config)

    if "signal_assets_configuration" not in portfolio_build_config['backtesting_weights_configuration']['signal_weights_configuration']:
        if not auto_complete:
            raise Exception(
                "Missing 'signal_weights_configuration' in 'backtesting_weights_configuration'"
            )
        logger.warning("signal_weights_configuration missing asset configuration copying from portfolio")
        portfolio_build_config['backtesting_weights_configuration']['signal_weights_configuration']["signal_assets_configuration"]= portfolio_build_config["assets_configuration"]
    else:
        swc__asset_universe_configuration = \
        portfolio_build_config['backtesting_weights_configuration']['signal_weights_configuration'][
            'signal_assets_configuration']['asset_universe']
        portfolio_build_config['backtesting_weights_configuration']['signal_weights_configuration'][
            'signal_assets_configuration']["asset_universe"] = AssetUniverse(**swc__asset_universe_configuration)

    configuration["portfolio_vam_configuration"]['front_end_details'] = configuration["portfolio_vam_configuration"]['front_end_details']

    if "portfolio_prices_frequency" not in portfolio_build_config:
        if not auto_complete:
            raise Exception("Missing 'portfolio_prices_frequency' in 'portfolio_build_config'")

        portfolio_build_config["portfolio_prices_frequency"] = "1days"
        logger.warning("Missing 'portfolio_prices_frequency' in 'portfolio_build_config' - Added default 1days")

    return PortfolioConfiguration.parse_portfolio_configurations(
        portfolio_build_configuration=portfolio_build_config,
        portfolio_vam_configuration=configuration['portfolio_vam_configuration'],
        portfolio_tdag_update_configuration=configuration['portfolio_tdag_update_configuration'],
    )


class TemplateFactory:
    """
    A factory for creating template-based objects, for example, market indices.
    """

    @staticmethod
    def create_market_index(index_name):
        """
        Creates a market index portfolio object based on a predefined template configuration.

        Args:
            index_name (str): The name of the index to create, which corresponds to a specific template configuration.

        Returns:
            PortfolioStrategy: A PortfolioStrategy object configured according to the template.
        """
        from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
        from .portfolio_templates.crypto_index_template import crypto_index_template_config

        logger.info(f"Creating market index for {index_name}")

        try:
            base_config = crypto_index_template_config
            if index_name == "CryptoTop5":
                base_config["backtesting_weights_config"]["signal_weights_configuration"]["num_top_assets"] = 5

            template_portfolio = PortfolioInterface(portfolio_config=base_config)
            return template_portfolio.portfolio_strategy  # Return first node of the portfolio
        except KeyError as e:
            logger.error(f"Configuration for {index_name} is missing: {e}")
            raise
