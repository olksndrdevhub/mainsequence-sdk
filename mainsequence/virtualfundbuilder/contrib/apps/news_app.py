#!/usr/bin/env python3
import os
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import plotly.graph_objs as go
# Assuming TDAGAgent is correctly set up and accessible in the execution environment
from mainsequence.virtualfundbuilder.agent_interface import TDAGAgent
from pydantic import BaseModel, Field
from polygon import RESTClient
import dotenv
# Jinja2 for report generation
from jinja2 import Environment, FileSystemLoader

from mainsequence.client import AssetCategory, Asset
from mainsequence.client.models_tdag import Artifact
from mainsequence.virtualfundbuilder.resource_factory.app_factory import BaseApp

# --- Configuration and Setup ---

dotenv.load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

if not POLYGON_API_KEY:
    print("Warning: POLYGON_API_KEY environment variable not set. Data fetching will fail.")

class SentimentReportConfig(BaseModel):
    """Pydantic model defining parameters for the Sentiment Report."""
    asset_category: str = "top_10_us_equity_market_cap"
    report_days: int = 14
    report_title: str = "Multi-Ticker News Sentiment & Headlines Report"
    bucket_name: str = "SentimentReports" # Optional: For artifact storage
    authors: str = "Automated Analysis (Main Sequence AI)"
    sector: str = "Technology Focus"
    region: str = "Global"
    topics: List[str] = Field(default_factory=lambda: ["Sentiment Analysis", "News Aggregation", "Market Data", "Equities"])
    news_items_per_day_limit: int = 5
    report_id: Optional[str] = "MS_SentimentReport"


class SentimentReport(BaseApp):
    """
    Generates an HTML report summarizing news sentiment and headlines
    for a list of stock tickers using data from Polygon.io.
    """
    configuration_class = SentimentReportConfig

    def __init__(self, configuration: SentimentReportConfig):
        self.configuration = configuration
        self.tdag_agent = TDAGAgent()

        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=self.configuration.report_days -1)
        self.start_date = start_date_dt.strftime("%Y-%m-%d")
        self.end_date = end_date_dt.strftime("%Y-%m-%d")

        category = AssetCategory.get(unique_identifier=self.configuration.asset_category)
        self.tickers = [a.symbol for a in category.get_assets()]
        self.category_name = category.display_name
        # Initialize Polygon client once if API key exists
        self.polygon_client = RESTClient(POLYGON_API_KEY) if POLYGON_API_KEY else None
        # Setup Jinja2 environment once
        self._setup_jinja()

    def _setup_jinja(self):
        """Initializes the Jinja2 environment."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        if not os.path.isdir(template_dir):
            raise FileNotFoundError(f"Jinja2 template directory not found: {template_dir}")
        report_template_path = os.path.join(template_dir, "report.html")
        if not os.path.isfile(report_template_path):
             raise FileNotFoundError(f"Jinja2 report template not found: {report_template_path}")
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

    def _fetch_data(self) -> (Dict[str, pd.DataFrame], Dict[str, Dict[str, List[Dict]]]):
        """
        Fetches sentiment counts and news headlines for configured tickers and date range.
        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, List[Dict]]]]:
                - Sentiment data per ticker.
                - News items (title, url) per ticker per date.
        """
        if not self.polygon_client:
            print("Error: Polygon API key not configured. Cannot fetch data.")
            empty_sentiment = {ticker: pd.DataFrame() for ticker in self.tickers}
            empty_news = {ticker: {} for ticker in self.tickers}
            return empty_sentiment, empty_news

        tickers = self.tickers
        start_date = self.start_date
        end_date = self.end_date

        results = {}
        all_news = {}
        date_range = pd.date_range(start=start_date, end=end_date)

        print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}")

        for ticker in tickers:
            print(f" -> Fetching for {ticker}...")
            sentiment_count = []
            ticker_news_by_date = {}
            for day in date_range:
                day_str = day.strftime("%Y-%m-%d")
                try:
                    # Fetch news for the day
                    daily_news_response = list(
                        self.polygon_client.list_ticker_news(
                            ticker=ticker, published_utc=day_str, limit=100 # Limit news fetched per day
                        )
                    )
                except Exception as e:
                    print(f"    Error fetching news for {ticker} on {day_str}: {e}")
                    daily_news_response = []

                daily_sentiment = {"date": day, "positive": 0, "negative": 0, "neutral": 0}
                daily_news_items_for_report = [] # Store title/url for the report

                for article in daily_news_response:
                    # Extract headline and URL for the report list
                    if hasattr(article, 'title') and hasattr(article, 'article_url'):
                        daily_news_items_for_report.append({'title': article.title, 'url': article.article_url})

                    # Extract sentiment from insights
                    if hasattr(article, "insights") and article.insights:
                        for insight in article.insights:
                            if hasattr(insight, 'sentiment'):
                                sentiment = insight.sentiment
                                if sentiment == "positive": daily_sentiment["positive"] += 1
                                elif sentiment == "negative": daily_sentiment["negative"] += 1
                                elif sentiment == "neutral": daily_sentiment["neutral"] += 1

                sentiment_count.append(daily_sentiment)
                # Store the collected headlines/urls for this date
                if daily_news_items_for_report:
                    ticker_news_by_date[day_str] = daily_news_items_for_report

            if sentiment_count:
                df_sentiment = pd.DataFrame(sentiment_count)
                df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])
                df_sentiment.set_index("date", inplace=True)
                # Ensure all dates in the range are present, filling missing ones with 0
                df_sentiment = df_sentiment.reindex(date_range, fill_value=0)
                results[ticker] = df_sentiment
                all_news[ticker] = ticker_news_by_date # Store news items (title/url) for this ticker
            else:
                print(f"    No sentiment data found for {ticker} in the date range.")
                results[ticker] = pd.DataFrame(index=date_range, columns=['positive', 'negative', 'neutral']).fillna(0)
                all_news[ticker] = {} # No news items stored if no sentiment data

        return results, all_news

    def _generate_plot(self, df_sentiment: pd.DataFrame, chart_title: str) -> Optional[str]:
        """
        Generates a Plotly sentiment chart and returns it as a Base64 encoded PNG string.
        Returns None if no data to plot or if image generation fails.
        """
        if df_sentiment.empty or (df_sentiment[['positive', 'negative', 'neutral']].sum().sum() == 0):
             print(f"    No data to plot for '{chart_title}'. Skipping chart generation.")
             return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sentiment.index, y=df_sentiment["positive"], mode="lines+markers", name="Positive", line=dict(color="green"), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=df_sentiment.index, y=df_sentiment["negative"], mode="lines+markers", name="Negative", line=dict(color="red"), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=df_sentiment.index, y=df_sentiment["neutral"], mode="lines+markers", name="Neutral", line=dict(color="gray", dash="dash"), marker=dict(size=5)))

        fig.update_layout(
            title=f"{chart_title} News Sentiment Over Time",
            xaxis_title="Date", yaxis_title="Sentiment Count", legend_title="Sentiment",
            width=850, height=450, margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(tickformat="%Y-%m-%d")
        )

        buf = BytesIO()
        try:
            fig.write_image(buf, format="png", scale=2)
            buf.seek(0)
            encoded_plot = base64.b64encode(buf.read()).decode("utf-8")
            return encoded_plot
        except Exception as e:
            print(f"    Error generating PNG for '{chart_title}': {e}")
            # Ensure kaleido is installed: pip install kaleido
            return None

    def run(self) -> str:
        """
        Orchestrates the report generation process: fetch data, create plots, render HTML.
        Returns the path to the generated HTML file.
        """
        print(f"Running Sentiment Report with configuration: {self.configuration.report_id}")

        all_sentiment_data, all_news_data = self._fetch_data()

        valid_dfs = [df for df in all_sentiment_data.values() if not df.empty]
        combined_chart_base64 = None
        if valid_dfs:
            combined_df = pd.concat(valid_dfs).groupby(level=0).sum()
            combined_chart_base64 = self._generate_plot(combined_df, "All Tickers (Combined)")

        if combined_chart_base64:
            combined_chart_html = f"""
                <h2>Combined Sentiment Across All Tickers</h2>
                <p style="text-align:center;">
                    <img alt="All Tickers Combined Sentiment Chart"
                         src="data:image/png;base64,{combined_chart_base64}"
                         style="max-width:850px; width:100%; display: block; margin:auto;">
                </p><hr style="margin: 30px 0;">"""
        else:
            combined_chart_html = "<h2>Combined Sentiment</h2><p>No combined sentiment data available.</p><hr style='margin: 30px 0;'>"

        ticker_sections_html = ""
        print("\nGenerating individual ticker sections...")
        for ticker in self.tickers:
            df_sentiment = all_sentiment_data.get(ticker)
            # ticker_news contains { "YYYY-MM-DD": [ {'title': '...', 'url': '...'}, ... ], ... }
            ticker_news = all_news_data.get(ticker, {})
            ticker_html = f"<h2>{ticker} Sentiment & News</h2>\n"
            print(f" -> Processing {ticker}")

            chart_base64 = self._generate_plot(df_sentiment, ticker)
            if chart_base64:
                ticker_html += f"""
                    <div style="text-align: center; margin-bottom: 20px;">
                        <img alt="{ticker} Sentiment Chart" src="data:image/png;base64,{chart_base64}"
                             style="max-width:850px; width:100%; display: block; margin:auto;">
                    </div>"""
            else:
                ticker_html += f"<p>No plottable sentiment data available for {ticker}.</p>\n"

            # Aggregate headlines for summarization
            all_headlines_for_ticker = []
            if ticker_news:
                # Iterate through dates, then news items for that date
                for date_str in sorted(ticker_news.keys(), reverse=True): # Process recent dates first if needed
                    for item in ticker_news[date_str]:
                        all_headlines_for_ticker.append(item.get('title', '')) # Collect titles

            summarized_news_html = "" # Initialize summary HTML block
            if all_headlines_for_ticker:
                 # Join headlines into a single string for the agent
                 # Filter out empty strings just in case
                news_headlines_text = ". ".join(filter(None, all_headlines_for_ticker))

                # Important assumption: The Polygon API (`list_ticker_news`) does not provide
                # full news article text, only headlines and metadata.
                # Therefore, the summary is generated *based on headlines only*.
                if news_headlines_text:
                    print(f"    -> Generating summary for {ticker} based on {len(all_headlines_for_ticker)} headlines...")
                    try:
                        # Construct the prompt for the agent
                        summary_prompt = (
                            f"Review the following news headlines for {ticker} over the last "
                            f"{self.configuration.report_days} days and provide a concise summary "
                            f"(around 100-150 words) of the key themes or major news points mentioned. "
                            f"Headlines: {news_headlines_text}"
                        )
                        # Call the agent
                        summarized_news_text = self.tdag_agent.query_agent(summary_prompt)
                        # Format the summary for HTML report
                        # Basic paragraph format, can be enhanced
                        summarized_news_html = f"<h4>Headlines Summary (AI Generated)</h4><p>{summarized_news_text}</p>\n"
                        print(f"    -> Summary generated for {ticker}.")
                    except NameError:
                        print(f"    Error: TDAGAgent not defined or available. Skipping summary for {ticker}.")
                        summarized_news_html = "<h4>Headlines Summary (AI Generated)</h4><p>Summary generation failed: TDAGAgent not available.</p>\n"
                    except Exception as e:
                        print(f"    Error calling TDAGAgent for {ticker}: {e}")
                        summarized_news_html = f"<h4>Headlines Summary (AI Generated)</h4><p>Error generating summary: {e}</p>\n"
                else:
                    # Case where headlines were collected but resulted in an empty string
                     summarized_news_html = "<h4>Headlines Summary (AI Generated)</h4><p>No headline text available to generate a summary.</p>\n"
            else:
                # Case where no news items were found for the ticker at all
                summarized_news_html = "<h4>Headlines Summary (AI Generated)</h4><p>No news headlines found for this period to generate a summary.</p>\n"

            # Add the generated summary (or error message) to the ticker's HTML section
            ticker_html += summarized_news_html

            ticker_html += "<h3>Recent News Headlines (Details)</h3>\n" # Changed title slightly for clarity
            if ticker_news:
                sorted_dates = sorted(ticker_news.keys(), reverse=True)
                news_list_html = ""
                total_listed_count = 0
                for date_str in sorted_dates:
                    news_items = ticker_news[date_str]
                    if news_items:
                        # Limit per day applied here
                        items_to_show = news_items[:self.configuration.news_items_per_day_limit]
                        if items_to_show:
                            news_list_html += f"<h5>{date_str}</h5>\n<ul class='list-unstyled'>\n"
                            for item in items_to_show:
                                safe_title = item.get('title', 'No Title').replace('<', '&lt;').replace('>', '&gt;')
                                url = item.get('url', '#')
                                news_list_html += f"  <li><a href='{url}' target='_blank' rel='noopener noreferrer'>{safe_title}</a></li>\n"
                                total_listed_count += 1
                            news_list_html += "</ul>\n"

                # Add the detailed list or a message if nothing was listed
                ticker_html += news_list_html if news_list_html else "<p>No recent news headlines found or listed based on limits.</p>\n"
            else:
                ticker_html += "<p>No news headlines found for this period.</p>\n" # Message if ticker_news was empty

            # Add the horizontal rule after all content for the ticker
            ticker_sections_html += ticker_html + '<hr style="margin: 30px 0;">\n'

        report_content_html = f"""
            <h2>Overview</h2>
            <p>This report summarizes daily sentiment counts (positive/negative/neutral)
            derived from Polygon.io news article insights for each requested ticker,
            within the date range {self.start_date} to {self.end_date}.</p>
            {combined_chart_html}
            {ticker_sections_html}
        """

        template_context = {
            "report_title": self.configuration.report_title,
            "report_id": self.configuration.report_id,
            "current_date": datetime.now().strftime('%Y-%m-%d'),
            "authors": self.configuration.authors,
            "sector": self.configuration.sector,
            "region": self.configuration.region,
            "topics": self.configuration.topics,
            "current_year": datetime.now().year,
            "summary": (
                f"Daily sentiment analysis, AI-generated headline summary, and news headlines for the {self.category_name} category "
                f"from {self.start_date} to {self.end_date}. "
                "Includes combined and individual ticker views."
            ),
            "report_content": report_content_html,
            "logo_location": f"https://main-sequence.app/static/media/logos/MS_logo_long_white.png",
        }

        template = self.jinja_env.get_template("report.html")
        rendered_html = template.render(template_context)

        output_html_path = os.path.join(os.path.dirname(__file__), "multi_ticker_sentiment_report.html")
        try:
            with open(output_html_path, "w", encoding="utf-8") as f:
                f.write(rendered_html)
            print(f"\nHTML report generated successfully: {output_html_path}")
            print(f"View the report at: file://{os.path.abspath(output_html_path)}")
        except Exception as e:
            print(f"\nError writing HTML report to file: {e}")
            return None

        # Upload the generated HTML as an artifact
        html_artifact = None
        try:
             html_artifact = Artifact.upload_file(
                filepath=output_html_path,
                name=self.configuration.report_id + f"_{self.category_name}.html", # Added .html extension
                created_by_resource_name=self.__class__.__name__,
                bucket_name=self.configuration.bucket_name
            )
             print(f"Artifact uploaded successfully: {html_artifact.unique_identifier if html_artifact else 'Failed'}")
        except Exception as e:
            print(f"Error uploading artifact: {e}")

        return html_artifact


# --- Main Execution Guard ---
if __name__ == "__main__":
    try:
        import kaleido
    except ImportError:
        print("Warning: 'kaleido' package not found. Plotly image export might fail.")
        print("Consider installing it: pip install kaleido")

    # Example Configuration: Using Magnificent 7 for 7 days
    config = SentimentReportConfig(
        asset_category="magnificent_7", # Example category
        report_days=7,
        report_title="Magnificent 7 News Sentiment & Headlines Report (Last 7 Days)",
        report_id="Mag7_SentimentReport_7d" # More specific ID
    )

    # Create App instance with configuration
    app = SentimentReport(config)

    # Run the report generation and get the artifact object
    generated_artifact = app.run()

    if generated_artifact:
        print(f"\nReport generation complete. Artifact ID: {generated_artifact.id}")
    else:
        print("\nReport generation completed, but artifact upload failed or was skipped.")