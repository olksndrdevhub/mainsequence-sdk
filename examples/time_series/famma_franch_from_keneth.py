import os
import datetime
import pytz
import pandas as pd
from typing import Optional

import requests
import zipfile
import io
import dotenv

dotenv.load_dotenv('../../.env')  # Load environment variables from .env
from mainsequence.tdag import TimeSerie
from mainsequence.tdag_client.models import DataUpdates


class KennethFrenchTimeSerie(TimeSerie):
    """
    A TimeSerie that retrieves the daily Fama-French 3-factor data (Mkt-RF, SMB, HML)
    directly from the Kenneth French website.

    The raw file typically includes columns:
        Date (YYYYMMDD)
        Mkt-RF
        SMB
        HML
        RF
    in percent. We'll parse them, convert to decimal, and store them in a DataFrame.

    Steps in update:
      1. Use update_statistics to figure out how far back we need data.
      2. Download & extract the daily 3-factor CSV from Kenneth French's site.
      3. Parse & load the data into a DataFrame.
      4. Filter rows that are strictly after the last update date.
      5. Return that data with a multi-index: (time_index, unique_identifier) = (Date, 'KEN_FRENCH_3')
    """

    # Link to the zipped CSV for the daily Fama-French factors
    # e.g. http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/
    # for demonstration, we use the widely known link:
    FAMA_FRENCH_DAILY_ZIP_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

    SIM_OFFSET_START = datetime.timedelta(days=365 * 20)  # fallback 20 years

    @TimeSerie._post_init_routines()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, update_statistics: DataUpdates):
        # 1) Determine how far back to fetch data using update_statistics
        #    If empty, fallback to 20 years ago.
        now_utc = datetime.datetime.now(pytz.utc)
        update_statistics = update_statistics.update_assets(
            asset_list=None,
            init_fallback_date=now_utc - self.SIM_OFFSET_START
        )
        min_date = update_statistics.get_min_latest_value(init_fallback_date=now_utc - self.SIM_OFFSET_START)

        # We'll only keep factor observations strictly greater than min_date.
        # So if min_date is 2020-01-01, we only want data from 2020-01-02 onward.
        cutoff_date = (min_date + datetime.timedelta(days=1)).date()

        # 2) Download & extract the daily 3-factor CSV
        try:
            resp = requests.get(self.FAMA_FRENCH_DAILY_ZIP_URL, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"Failed to download Fama-French data: {e}")
            return pd.DataFrame()

        # The content is a ZIP file containing a single CSV, typically named 'F-F_Research_Data_Factors_daily.CSV'
        zip_content = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = zip_content.namelist()[0]  # There's typically only one file in the zip
        with zip_content.open(csv_name) as f:
            raw_bytes = f.read()
        # Convert bytes to text
        raw_text = raw_bytes.decode("utf-8")

        # 3) Parse the CSV. The file structure includes:
        #    header lines, the data table, then lines after 'END'. We skip non-data lines.

        data_lines = []
        lines = raw_text.splitlines()
        collecting = False
        for line in lines:
            if "Observations:" in line:
                # typically we skip the last lines after Observations or after the line 'END'
                break
            if collecting and line.strip():
                data_lines.append(line)
            if "" in line and not collecting:
                # There's often a blank line before data starts, but it's not always consistent.
                # Another approach is to look for a line that starts with 'Date' or the first data row.
                # We'll just keep collecting from the first numeric line until we see 'END'.
                pass
            # Another approach: If the line starts with a 6 or 8-digit number, we consider it data.
            if not collecting:
                # try to detect the first data row:
                parts = line.split(",")
                if len(parts) >= 4:
                    # check if first is numeric
                    if parts[0].strip().isdigit():
                        collecting = True
                        data_lines.append(line)

        if not data_lines:
            print("No data lines found in the Fama-French file.")
            return pd.DataFrame()

        # We'll create a small CSV out of data_lines.
        # The daily file format typically looks like:
        # 19260701, -2.38,   0.41,   0.06, 0.010   # Mkt-RF, SMB, HML, RF in that order.
        # or 19260701, -2.38, 0.41, 0.06, 0.010
        # We'll parse them into a DataFrame with columns: [Date, Mkt-RF, SMB, HML, RF].

        # Some lines might be missing columns or have extra whitespace.
        # We'll parse them carefully.

        records = []
        for line in data_lines:
            # example: 19260701,  -2.21,   0.11,   0.15,  0.010
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            try:
                date_str = parts[0]
                # If there's an extra 5th col for RF, parse it. If not, it's in col 4.
                # The daily dataset typically has 5 columns: date, Mkt-RF, SMB, HML, RF
                # but sometimes the last column might be missing.
                # We'll assume 5 columns are present.
                mktrf = float(parts[1])
                smb = float(parts[2])
                hml = float(parts[3])
                rf = 0.0
                if len(parts) >= 5:
                    rf = float(parts[4])
                # Convert date_str from YYYYMMDD => date.
                dt = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                records.append((dt, mktrf, smb, hml, rf))
            except:
                continue

        df = pd.DataFrame(records, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
        if df.empty:
            print("Parsed DF from Ken French data is empty.")
            return pd.DataFrame()

        # Convert from percent => decimal.
        # e.g. -2.38 => -0.0238
        for col in ["Mkt-RF", "SMB", "HML", "RF"]:
            df[col] = df[col] / 100.0

        # 4) Filter rows strictly after cutoff_date.
        df = df[df["date"] > cutoff_date]
        if df.empty:
            print("No new data after cutoff.")
            return pd.DataFrame()

        # 5) Create a time_index with daily frequency
        #    We'll store Mkt-RF, SMB, HML, & possibly we keep RF separate
        #    But for the 3-factor model, we typically only keep Mkt-RF, SMB, HML.
        df.rename(columns={"Mkt-RF": "MKT_RF"}, inplace=True)
        # Convert to a standard datetime
        df["time_index"] = pd.to_datetime(df["date"])

        # We'll keep the columns as MKT_RF, SMB, HML.
        # If needed, you can incorporate the risk-free separately.
        factor_df = df[["time_index", "MKT_RF", "SMB", "HML","RF"]].copy()

        # 6) Return as a multi-index (time_index, 'KEN_FRENCH_3')
        factor_df["unique_identifier"] = "KEN_FRENCH_3"
        factor_df.set_index(["time_index", "unique_identifier"], inplace=True)



        return factor_df


###############################################
# Example usage / Test Script
###############################################
def test_kenneth_french_time_serie():
    kfts = KennethFrenchTimeSerie()

    # First run with empty updates => fetch full data up to 20 yrs back
    print("=== FIRST RUN (EMPTY DataUpdates) ===")
    df1 = kfts.update(DataUpdates())
    print(df1)

    if not df1.empty:
        # We store the max date as the last update
        update_dict = (
            df1.reset_index().groupby("unique_identifier")["time_index"].max().to_dict()
        )
        updates = DataUpdates(update_statistics=update_dict,
                              max_time_index_value=df1.index.get_level_values("time_index").max())
    else:
        updates = DataUpdates()

    # Second run => incremental from the last date
    print("\n=== SECOND RUN (INCREMENTAL) ===")
    df2 = kfts.update(updates)
    print(df2)


if __name__ == "__main__":
    test_kenneth_french_time_serie()
