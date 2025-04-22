#!/usr/bin/env python3
import os
from datetime import datetime, timedelta
from typing import List, Optional, Union

import pandas as pd
import pytz
from mainsequence.virtualfundbuilder.utils import get_vfb_logger, TIMEDELTA
from mainsequence.virtualfundbuilder.resource_factory.signal_factory import register_signal_class, WeightsBase
from mainsequence.client import AssetCategory, Asset, DataUpdates
from pydantic import BaseModel, Field
from polygon import RESTClient
from mainsequence.tdag.time_series import TimeSerie
from tqdm import tqdm # For progress bar

logger = get_vfb_logger()

@register_signal_class()
class SentimentSignal(WeightsBase, TimeSerie):
    """
    Calculates portfolio weights based on aggregated news sentiment from Polygon.io.
    Weights are proportional to a score calculated as (Positive Articles - Negative Articles).
    Assets with insufficient recent news sentiment are excluded.
    This class acts as a TimeSerie node providing daily sentiment-based signal weights,
    starting from OFFSET_START on the first run.
    """
    OFFSET_START = datetime.now() - timedelta(days=90)
    @TimeSerie._post_init_routines()
    def __init__(self, min_articles_threshold: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
        self.polygon_client = RESTClient(POLYGON_API_KEY)
        self.min_articles_threshold = min_articles_threshold
        self._fetch_assets()

    def _fetch_assets(self):
        """Fetches assets from the specified category."""
        try:
            category = AssetCategory.get(unique_identifier=self.assets_configuration.assets_category_unique_id)
            self.assets: List[Asset] = category.get_assets()
            self.asset_map = {asset.ticker: asset.unique_identifier for asset in self.assets}
            self.tickers = list(self.asset_map.keys())
            logger.info(
                f"SentimentSignal: Fetched {len(self.tickers)} assets for category '{self.assets_configuration.assets_category_unique_id}'.")
        except Exception as e:
            logger.exception(
                f"Failed to fetch assets for category {self.assets_configuration.assets_category_unique_id}")
            raise e

    def get_explanation(self):
         return f"""
         <p>{self.__class__.__name__}: Signal calculates weights based on aggregated news sentiment from Polygon.io.</p>
         <p>Requires at least <b>{self.min_articles_threshold}</b> articles with sentiment for an asset to be included.</p>
         <p>Weights are proportional to the sentiment score (Positive Articles - Negative Articles), normalized across assets with positive scores.</p>
         """

    def maximum_forward_fill(self) -> timedelta:
         """Signal is daily, so valid for slightly less than a day."""
         return timedelta(days=1) - TIMEDELTA

    def _fetch_sentiment_data(self, news_date: datetime) -> pd.DataFrame:
        """Fetches and aggregates sentiment for the lookback period *ending* on end_date_of_lookback."""
        all_sentiment_data = []

        if not hasattr(self, 'tickers') or not self.tickers:
             self._fetch_assets()
        if not self.tickers:
             logger.warning("No tickers available to fetch sentiment for.")
             return pd.DataFrame()

        for ticker in self.tickers:
            ticker_sentiment = {"ticker": ticker, "positive": 0, "negative": 0, "neutral": 0, "total_articles": 0}
            # Fetch for the entire range at once if API allows, otherwise loop day by day
            try:
                day_str = news_date.strftime("%Y-%m-%d")
                news_response = list(
                    self.polygon_client.list_ticker_news(
                        ticker=ticker, published_utc=day_str, limit=100 # Limit news fetched per day
                    )
                )
                for article in news_response:
                     if hasattr(article, "insights") and article.insights:
                         ticker_sentiment["total_articles"] += 1
                         for insight in article.insights:
                             if hasattr(insight, 'sentiment'):
                                 sentiment = insight.sentiment
                                 if sentiment == "positive": ticker_sentiment["positive"] += 1
                                 elif sentiment == "negative": ticker_sentiment["negative"] += 1
                                 elif sentiment == "neutral": ticker_sentiment["neutral"] += 1

            except Exception as e:
                logger.warning(f"Error fetching news for {ticker} from {day_str}: {e}")

            all_sentiment_data.append(ticker_sentiment)

        if not all_sentiment_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_sentiment_data)
        return df.set_index("ticker")

    def _calculate_weights_for_date(self, df_sentiment: pd.DataFrame) -> pd.Series:
        """Calculates normalized weights from aggregated sentiment scores."""
        if df_sentiment.empty:
            return pd.Series(dtype=float)

        # Filter assets based on minimum articles threshold
        df_sentiment_filtered = df_sentiment[df_sentiment["total_articles"] >= self.min_articles_threshold].copy() # Use copy

        if df_sentiment_filtered.empty:
            logger.debug(f"No assets met the minimum article threshold ({self.min_articles_threshold}).")
            return pd.Series(dtype=float)

        # Calculate sentiment score
        df_sentiment_filtered["score"] = df_sentiment_filtered["positive"] - df_sentiment_filtered["negative"]

        # Calculate weights - proportional to score, but only positive scores allowed
        df_sentiment_filtered["raw_weight"] = df_sentiment_filtered["score"].clip(lower=0)
        total_positive_score = df_sentiment_filtered["raw_weight"].sum()

        if total_positive_score <= 0:
            logger.debug("Total positive sentiment score is zero or negative. Assigning zero weights.")
            df_sentiment_filtered["signal_weight"] = 0.0
        else:
            df_sentiment_filtered["signal_weight"] = df_sentiment_filtered["raw_weight"] / total_positive_score

        # Return only the weights series, indexed by ticker
        return df_sentiment_filtered["signal_weight"]


    def update(self, update_statistics: DataUpdates) -> pd.DataFrame:
        """
        Calculates daily sentiment weights for the required date range.
        """
        # --- Determine Date Range ---
        if update_statistics.max_time_index_value is None:
            # First run: start from OFFSET_START
            start_calc_date = self.OFFSET_START.date()
            logger.info(f"First run, starting calculation from OFFSET_START: {start_calc_date}")
        else:
            # Subsequent runs: start from the day after the last update
            start_calc_date = (update_statistics.max_time_index_value + timedelta(days=1)).date()
            logger.info(f"Subsequent run, starting calculation from: {start_calc_date}")

        # Calculate up to yesterday UTC
        end_calc_date = datetime.now(pytz.utc).date() - timedelta(days=1)

        if start_calc_date > end_calc_date:
            logger.info(f"Start date {start_calc_date} is after end date {end_calc_date}. No new dates to process.")
            return pd.DataFrame()

        update_days_per_run = 30
        if end_calc_date - start_calc_date > timedelta(days=update_days_per_run):
            end_calc_date = start_calc_date + timedelta(days=update_days_per_run)

        # --- Iterate and Calculate Weights ---
        all_daily_weights = []
        date_range = pd.date_range(start=start_calc_date, end=end_calc_date, freq='D')

        logger.info(f"Calculating sentiment weights from {start_calc_date} to {end_calc_date}...")
        for target_date in tqdm(date_range, desc="Calculating Daily Sentiment Weights"):
            # Fetch sentiment data for the lookback period ending on target_date
            df_sentiment_agg = self._fetch_sentiment_data(target_date)

            # Calculate weights for this specific target_date
            daily_weights = self._calculate_weights_for_date(df_sentiment_agg)

            if not daily_weights.empty:
                # Add the date as the time_index
                daily_weights_df = daily_weights.to_frame(name="signal_weight")
                daily_weights_df["time_index"] = pd.Timestamp(target_date, tz=pytz.utc) # Assign the date as timestamp
                all_daily_weights.append(daily_weights_df)

        if not all_daily_weights:
            logger.warning("No weights were calculated for the specified date range.")
            return pd.DataFrame()

        # --- Combine and Format Results ---
        df_combined = pd.concat(all_daily_weights)
        df_combined = df_combined.reset_index() # Get ticker into columns

        # Map ticker back to unique_identifier
        df_combined["unique_identifier"] = df_combined["ticker"].map(self.asset_map)
        df_combined = df_combined.dropna(subset=["unique_identifier"])

        if df_combined.empty:
            logger.warning("Output DataFrame is empty after mapping tickers. Returning empty weights.")
            return pd.DataFrame()

        # Set the final multi-index
        df_output = df_combined.set_index(["time_index", "unique_identifier"])[["signal_weight"]]

        logger.info(f"Generated {len(df_output)} total sentiment weight entries.")
        logger.debug(f"Final weights sample:\n{df_output.head()}")

        return df_output