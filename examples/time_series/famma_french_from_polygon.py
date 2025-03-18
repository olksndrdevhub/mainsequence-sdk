import os
import datetime
import pytz
import pandas as pd
from typing import Optional
import dotenv

dotenv.load_dotenv('../../.env')  # Load environment variables from .env

from polygon import RESTClient
from mainsequence.tdag import TimeSerie
from mainsequence.mainsequence_client import DataUpdates



class FamaFrenchTimeSerie(TimeSerie):
    """
    This TimeSerie dynamically fetches a cross-section of U.S. stocks from the Polygon API, calculates
    daily returns, sorts them by size (market cap) and by value (book-to-market) to replicate the
    Fama-French 3-Factor model:
        - Market risk premium (MKT-RF)
        - Small Minus Big (SMB)
        - High Minus Low (HML)

    Key Steps:
        1. Query Polygon for an equity universe (list of tickers) and relevant fundamental data.
        2. Download historical daily pricing for each ticker.
        3. Compute daily returns, market cap, and an approximate book-to-market ratio.
        4. Classify stocks into Small vs. Big (for SMB) and High vs. Low B/M (for HML).
        5. Compute daily factor returns.
    """
    SIM_OFFSET_START = datetime.timedelta(days=365*10)  # 1 year fallback

    @TimeSerie._post_init_routines()
    def __init__(self, *args, **kwargs):
        # No asset_list is required, we dynamically discover equities from Polygon.
        super().__init__(*args, **kwargs)

    def update(self, update_statistics: DataUpdates):
        """
        1) Determine how far back to fetch data using update_statistics.
        2) Pull a list of stocks from Polygon's reference endpoint.
        3) For each stock, fetch daily bars (OHLC), compute daily returns.
        4) Fetch or approximate fundamentals (e.g., shares outstanding, book value) to classify by size & B/M.
        5) Replicate standard Fama-French factor construction:
            - MKT = Value-weighted return of the entire universe.
            - SMB = Return(small stocks) - Return(big stocks).
            - HML = Return(high B/M) - Return(low B/M).
          For demonstration, we skip the exact Fama-French monthly rebal scheme and do a simplified daily approach.
        6) Return a DataFrame with index=(time_index, 'FAMAFRENC3') and columns=[MKT_RF, SMB, HML].
        """
        polygon_api_key = os.environ.get("POLYGON_API_KEY", "")
        if not polygon_api_key:
            print("Warning: POLYGON_API_KEY not found in environment.")

        now_utc = datetime.datetime.now(pytz.utc)

        # 1) Determine the date range using update_statistics
        #    If empty, fallback to (now - SIM_OFFSET_START)

        update_statistics = update_statistics.update_assets(
            asset_list=None,  # no direct assets
            init_fallback_date=now_utc - self.SIM_OFFSET_START
        )

        # We'll fetch from the earliest last_update among any tracked identifiers
        # but since we have no prior assets, let's just use a single fallback.
        min_date = update_statistics.get_min_latest_value(init_fallback_date=now_utc - self.SIM_OFFSET_START)
        start_date_str = (min_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        end_date_str = now_utc.strftime('%Y-%m-%d')

        # 2) Pull the equity universe from Polygon (reference/tickers) with pagination.
        client = RESTClient(polygon_api_key)
        ticker_universe = []
        page_cursor = None
        has_more = True

        DEBUG_LIMIT = 2

        while has_more:
            try:
                resp = client.list_tickers(
                    type="CS",
                    market="stocks",
                    limit=DEBUG_LIMIT,
                )
            except Exception as e:
                print(f"Polygon reference API call failed: {e}")
                return pd.DataFrame()

            count_on_page = 0
            for ticker_data in resp:
                ticker_universe.append(
                    {
                        "ticker": ticker_data.ticker,
                        # Possibly fetch fundamentals, shares_outstanding, etc.
                        "shares_outstanding": None,  # we will approximate below
                        "book_value": None,  # we will approximate below
                    }
                )
                count_on_page += 1

            # Attempt to advance the cursor if there's a next page.
            has_more = resp.has_next_page if hasattr(resp, 'has_next_page') else False
            page_cursor = resp.next_page if hasattr(resp, 'next_page') and has_more else None

            # Example early cutoff for demonstration:
            if len(ticker_universe) >= 5000:
                # If you want all, remove this guard.
                break
            # If no results, no need to continue.
            if count_on_page == 0:
                break

        if not ticker_universe:
            print("No tickers found!")
            return pd.DataFrame()

        # 3) For each ticker, fetch daily bars & compute daily returns.
        df_list = []
        for stock in ticker_universe:
            ticker = stock["ticker"]
            try:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date_str,
                    to=end_date_str,
                    limit=50000,
                )
                bars = pd.DataFrame(aggs["results"]) if "results" in aggs else pd.DataFrame()
                if bars.empty:
                    continue
                bars["time_index"] = pd.to_datetime(bars["t"], unit="ms")
                bars["ticker"] = ticker
                bars["adj_close"] = bars["c"]  # ignoring splits & dividends for brevity
                bars = bars.sort_values("time_index").reset_index(drop=True)

                # daily return
                bars["ret"] = bars["adj_close"].pct_change()
                df_list.append(bars[["time_index", "ticker", "adj_close", "ret"]])
            except Exception as e:
                print(f"Failed to fetch daily bars for {ticker}: {e}")

        if not df_list:
            print("No daily bars were retrieved.")
            return pd.DataFrame()

        full_prices = pd.concat(df_list, axis=0)
        full_prices.dropna(subset=["ret"], inplace=True)

        # 4) Approximate fundamentals for size & B/M classification.
        #    For demonstration, let's create random market cap & book-to-market.
        #    In production, you'd fetch real fundamentals from Polygon's financials endpoint.
        rng = pd.DataFrame(full_prices["ticker"].unique(), columns=["ticker"])
        rng["market_cap"] = rng.index * 1e8 + 5e8  # Mock: steadily increasing market caps
        rng["book_to_market"] = 0.5 + (rng.index / len(rng.index))  # Mock B/M factor
        fundamentals_map = rng.set_index("ticker").to_dict(orient="index")

        # Merge the fundamentals into full_prices.
        full_prices["market_cap"] = full_prices["ticker"].apply(
            lambda x: fundamentals_map[x]["market_cap"] if x in fundamentals_map else 1e9
        )
        full_prices["book_to_market"] = full_prices["ticker"].apply(
            lambda x: fundamentals_map[x]["book_to_market"] if x in fundamentals_map else 0.5
        )

        # 5) Group by day, compute factor returns.
        #    We'll define:
        #      - MKT: Value-weighted return of all stocks (simplified, ignoring risk-free).
        #      - SMB: Average return of small stocks - big stocks, using median market cap as the break.
        #      - HML: Average return of high B/M - low B/M, using median B/M as the break.

        def calc_daily_factors(subdf: pd.DataFrame) -> pd.Series:
            # We typically do (MKT minus risk-free). For example, we can assume RF=0 or fetch T-Bill data.
            # Here, we assume RF=0 for brevity.

            # Value-weighted MKT
            total_cap = subdf["market_cap"].sum()
            if total_cap == 0:
                mkt = 0.0
            else:
                w = subdf["market_cap"] / total_cap
                mkt = (w * subdf["ret"]).sum()

            # SMB
            size_median = subdf["market_cap"].median()
            small_stocks = subdf[subdf["market_cap"] <= size_median]
            big_stocks = subdf[subdf["market_cap"] > size_median]

            smb = small_stocks["ret"].mean() - big_stocks["ret"].mean()

            # HML
            bm_median = subdf["book_to_market"].median()
            high_bm = subdf[subdf["book_to_market"] > bm_median]
            low_bm = subdf[subdf["book_to_market"] <= bm_median]

            hml = high_bm["ret"].mean() - low_bm["ret"].mean()

            return pd.Series({"MKT_RF": mkt, "SMB": smb, "HML": hml})

        factor_df = (
            full_prices
            .groupby("time_index", as_index=True)
            .apply(calc_daily_factors)
        )

        # 6) Return a DataFrame with a multi-index (time_index, unique_identifier).
        factor_df.reset_index(drop=False, inplace=True)
        # We'll store factors under one pseudo-identifier: 'FAMA_FRENCH_3'
        factor_df["unique_identifier"] = "FAMA_FRENCH_3"
        factor_df.set_index(["time_index", "unique_identifier"], inplace=True)

        return factor_df

###############################################
# Example usage / Test Script
###############################################
def test_fama_french_time_serie():
    # 1) Create the TimeSerie instance
    ff_ts = FamaFrenchTimeSerie()

    # 2) CASE 1: Simulation with empty DataUpdates
    print("=== FIRST RUN WITH EMPTY DATA UPDATES ===")
    data_df = ff_ts.update(DataUpdates())
    print(data_df)

    # 3) Extract the max time per unique_identifier
    #    We'll store it in a new DataUpdates object.
    if not data_df.empty:
        update_dict = (
            data_df.reset_index().groupby("unique_identifier")["time_index"].max().to_dict()
        )
        updates = DataUpdates(update_statistics=update_dict,
                              max_time_index_value=data_df.index.get_level_values("time_index").max())
    else:
        updates = DataUpdates()

    # 4) CASE 2: Another run, simulating an incremental update.
    print("\n=== SECOND RUN USING EXTRACTED update_statistics ===")
    df_updates = ff_ts.update(updates)
    print(df_updates)

if __name__ == "__main__":
    test_fama_french_time_serie()
