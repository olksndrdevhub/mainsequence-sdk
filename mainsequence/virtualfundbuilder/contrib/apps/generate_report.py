#!/usr/bin/env python3

import base64
import os
from datetime import datetime
from io import BytesIO
from typing import List

from jinja2 import Environment, FileSystemLoader
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from pydantic import BaseModel
from jinja2 import Template


from mainsequence.client.models_tdag import Artifact
from mainsequence.virtualfundbuilder.resource_factory.app_factory import register_app, BaseApp


def example_data():
    """
    Generates two Plotly figures (line chart + correlation heatmap) from
    a fabricated 'fundamentals' DataFrame, returning their Base64-encoded PNGs.
    In a real setting, replace this function to fetch real data from the pylong API.
    """

    # ------------------------------------------------------------------------------
    # 1) EXAMPLE DATA
    #    The index is (datetime, asset_id); columns might include "Revenue" and "EPS".
    #    Here, we just fabricate random data for demonstration.

    dates = pd.date_range("2024-12-01", periods=5, freq="M")
    asset_ids = ["AAPL", "GOOG", "MSFT", "AMZN"]
    index = pd.MultiIndex.from_product([dates, asset_ids], names=["date", "asset_id"])

    np.random.seed(42)
    df_fundamentals = pd.DataFrame({
        "Revenue": np.random.randint(80, 120, len(index)),  # billions
        "EPS": np.random.rand(len(index)) * 5,  # earnings per share
    }, index=index).reset_index()

    # ------------------------------------------------------------------------------
    # 2) TIME-SERIES LINE CHART
    fig_line = px.line(
        df_fundamentals,
        x="date",
        y="Revenue",
        color="asset_id",
        title="Revenue Over Time by Asset"
    )

    # ------------------------------------------------------------------------------
    # 3) CORRELATION HEATMAP
    latest_date = df_fundamentals["date"].max()
    df_latest = df_fundamentals[df_fundamentals["date"] == latest_date]

    # Pivot so each row is an asset, columns are fundamentals:
    df_pivot = df_latest.pivot_table(index="asset_id", columns=None)[["Revenue", "EPS"]]

    corr_matrix = df_pivot.corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="Blues"
    ))
    fig_heatmap.update_layout(
        title=f"Correlation of Fundamentals on {latest_date.strftime('%Y-%m-%d')}"
    )

    # ------------------------------------------------------------------------------
    # 4) CONVERT PLOTS TO BASE64 STRINGS
    def fig_to_base64(fig):
        """Render a Plotly figure to a PNG and return a base64 string."""
        buf = BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    chart1_base64 = fig_to_base64(fig_line)
    chart2_base64 = fig_to_base64(fig_heatmap)

    return chart1_base64, chart2_base64


class ReportConfig(BaseModel):
    """Pydantic model defining the parameters for report generation."""
    report_id: str = "MC-2025"
    report_title: str = "Global Strategy Views: Diversify to Amplify"
    bucket_name: str = "Reports"
    authors: str = "Main Sequence AI"
    sector: str = "US Equities"
    region: str = "USA"
    topics: List[str] = ["Diversification", "Equities", "Fundamentals"]
    summary: str = (
        "We are entering a more benign phase of the economic cycle characterized by "
        "sustained economic growth and declining policy interest rates. Historically, "
        "such an environment supports equities but also highlights the increasing "
        "importance of broad diversification across regions and sectors."
    )

@register_app()
class ReportApp(BaseApp):
    """
    Minimal example of a 'ReportApp' that can:
    1) Generate dummy data and create charts (line + heatmap).
    2) Embed those charts into an HTML template.
    3) Optionally export the HTML to PDF using WeasyPrint.
    """
    configuration_class = ReportConfig
    def __init__(self, configuration: ReportConfig):
        self.configuration = configuration

    def _fig_to_base64(self, fig) -> str:
        """
        Render a Plotly figure to PNG and return a Base64 string.
        """
        buf = BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def run(self):
        """
        Generates an HTML report (and optional PDF) in a minimal, self-contained way.
        """
        print(f"Running tool with configuration {self.configuration}")

        # 1) Retrieve the chart images:
        chart1_base64, chart2_base64 = example_data()

        # Build context from config
        template_context = {
            "current_date": datetime.now().strftime('%Y-%m-%d'),
            "current_year": datetime.now().year,
            "logo_location": f"https://main-sequence.app/static/media/logos/MS_logo_long_white.png",
            # Pulling fields from our pydantic config:
            "report_id": self.configuration.report_id,
            "authors": self.configuration.authors,
            "sector": self.configuration.sector,
            "region": self.configuration.region,
            "topics": self.configuration.topics,
            "report_title": self.configuration.report_title,
            "summary": self.configuration.summary,
            "report_content": f"""
                <h2>Overview</h2>
                <p>
                    Longer-term interest rates are expected to remain elevated, driven by rising government deficits
                    and persistent term premiums. However, the reduced likelihood of a near-term recession presents
                    opportunities for positive equity returns, notably in sectors like technology and select value-oriented
                    areas such as financials.
                </p>
                <p>
                    This evolving landscape emphasizes the necessity of expanding our investment horizon beyond traditional
                    focuses—such as large-cap US technology—to include regional markets, mid-cap companies, and "Ex-Tech Compounders."
                    Such diversification aims to enhance risk-adjusted returns as global growth trajectories become more aligned.
                </p>
                <h2>Key Takeaways</h2>
                <ul>
                    <li>
                        <strong>Diversification enhances return potential:</strong> Capturing alpha in the upcoming cycle
                        will likely depend on a diversified approach across multiple regions and investment factors.
                    </li>
                    <li>
                        <strong>Technology remains essential:</strong> Rising demand for physical infrastructure, such as
                        data centers and AI-supportive hardware, will benefit traditional industrial sectors, creating
                        new investment opportunities.
                    </li>
                    <li>
                        <strong>Divergent interest rate dynamics:</strong> Central banks have started easing policies, but
                        persistent high bond yields imply limitations on further equity valuation expansions.
                    </li>
                </ul>

                <!-- Page break before next section if printing to PDF -->
                <div style="page-break-after: always;"></div>

                <h2>Fundamental Trends and Correlation Analysis</h2>
                <p>
                    The following charts illustrate recent fundamental trends among selected US equities, focusing specifically
                    on revenue performance over recent reporting periods. This analysis leverages data obtained via the internal
                    "pylong" API, clearly highlighting the evolving top-line dynamics across multiple companies.
                </p>
                <p style="text-align:center;">
                    <img alt="Revenue Over Time"
                         src="data:image/png;base64,{chart1_base64}"
                         style="max-width:600px; width:100%;">
                </p>

                <p>
                    Further, the correlation heatmap below illustrates the relationships between key fundamental indicators—such
                    as Revenue and Earnings Per Share (EPS)—across companies at the latest reporting date. This visualization
                    provides strategic insights into how closely fundamental metrics move together, enabling more informed
                    portfolio allocation decisions.
                </p>
                <p style="text-align:center;">
                    <img alt="Correlation Heatmap"
                         src="data:image/png;base64,{chart2_base64}"
                         style="max-width:600px; width:100%;">
                </p>

                <p>
                    As the macroeconomic environment evolves, shifts in these fundamental correlations may offer additional
                    opportunities for strategic repositioning and optimized sector exposures.
                </p>
            """
        }

        """
        Renders a static HTML report from Jinja2 templates, embedding two charts as Base64 images,
        and (optionally) saves it as PDF using WeasyPrint.
        """
        # 2) Setup the Jinja2 environment: (point to the templates directory)
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)

        # 3) Load the derived template (which should reference placeholders in Jinja syntax).
        template = env.get_template('report.html')

        # 5) Render the HTML:
        rendered_html = template.render(template_context)

        # 6) Write the rendered HTML to a file
        output_html_path = os.path.join(os.path.dirname(__file__), 'output_report.html')
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)

        print(f"HTML report generated: {output_html_path}")

        print("Generated HTML:", output_html_path)

        # from weasyprint import HTML
        # pdf_path = "/tmp/report.pdf"
        # HTML(string=rendered_html).write_pdf(pdf_path)
        # print(f"PDF generated: {pdf_path}")
        # pdf_artifact = Artifact.upload_file(filepath=pdf_path, name="Report PDF", created_by_resource_name=self.__class__.__name__, bucket_name="Reports")
        html_artifact = Artifact.upload_file(
            filepath=output_html_path,
            name=self.configuration.report_id,
            created_by_resource_name=self.__class__.__name__,
            bucket_name=self.configuration.bucket_name
        )
        return html_artifact

if __name__ == "__main__":
    # Example usage:
    config = ReportConfig()  # Or override fields as needed
    app = ReportApp(config)
    html_artifact = app.run()  # Creates output_report.html and weasy_output_report.pdf
