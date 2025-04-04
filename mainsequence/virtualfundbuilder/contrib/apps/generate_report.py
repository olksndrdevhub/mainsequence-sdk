#!/usr/bin/env python3

import base64
from datetime import datetime
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from pydantic import BaseModel
from jinja2 import Template


from mainsequence.client.models_tdag import Artifact
from mainsequence.virtualfundbuilder.resource_factory.app_factory import register_app, BaseApp


class ReportConfig(BaseModel):
    """Pydantic model defining the parameters for report generation."""
    report_id: str = "MC-2025-XYZ"
    authors: str = "Main Sequence AI"
    sector: str = "US Equities"
    region: str = "USA"
    topics: List[str] = ["Diversification", "Equities", "Fundamentals"]
    report_title: str = "Global Strategy Views: Diversify to Amplify"
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

    def _generate_charts(self):
        """
        Returns chart1_base64, chart2_base64 from fabricated data.
        For real usage, replace the data generation with your actual data.
        """
        # Dummy timeseries fundamentals data:
        dates = pd.date_range("2024-12-01", periods=5, freq="M")
        asset_ids = ["AAPL", "GOOG", "MSFT", "AMZN"]
        index = pd.MultiIndex.from_product([dates, asset_ids], names=["date", "asset_id"])

        np.random.seed(42)
        df_fundamentals = pd.DataFrame(
            {
                "Revenue": np.random.randint(80, 120, len(index)),  # billions
                "EPS": np.random.rand(len(index)) * 5,              # earnings per share
            },
            index=index
        ).reset_index()

        # 1) Time-Series Line Chart
        fig_line = px.line(
            df_fundamentals,
            x="date",
            y="Revenue",
            color="asset_id",
            title="Revenue Over Time by Asset"
        )

        # 2) Correlation Heatmap (latest date only)
        latest_date = df_fundamentals["date"].max()
        df_latest = df_fundamentals[df_fundamentals["date"] == latest_date]
        df_pivot = df_latest.pivot_table(index="asset_id")[["Revenue", "EPS"]]
        corr_matrix = df_pivot.corr()

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="Blues"
            )
        )
        fig_heatmap.update_layout(
            title=f"Correlation of Fundamentals on {latest_date.strftime('%Y-%m-%d')}"
        )

        chart1_base64 = self._fig_to_base64(fig_line)
        chart2_base64 = self._fig_to_base64(fig_heatmap)

        return chart1_base64, chart2_base64

    def run(self):
        """
        Generates an HTML report (and optional PDF) in a minimal, self-contained way.
        """
        print(f"Running tool with configuration {self.configuration}")

        # Create base64-encoded charts
        chart1, chart2 = self._generate_charts()

        # Build context from config
        context = {
            "current_date": datetime.now().strftime('%Y-%m-%d'),
            "current_year": datetime.now().year,
            "chart1": chart1,
            "chart2": chart2,
            # Pulling fields from our pydantic config:
            "report_id": self.configuration.report_id,
            "authors": self.configuration.authors,
            "sector": self.configuration.sector,
            "region": self.configuration.region,
            "topics": self.configuration.topics,
            "report_title": self.configuration.report_title,
            "summary": self.configuration.summary,
        }

        # For this minimal example, we embed the template here as a string:
        template_str = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ report_title }}</title>
</head>
<body>
    <h1>{{ report_title }}</h1>
    <p><strong>Report ID:</strong> {{ report_id }}</p>
    <p><strong>Authors:</strong> {{ authors }}</p>
    <p><strong>Sector:</strong> {{ sector }}</p>
    <p><strong>Region:</strong> {{ region }}</p>
    <p><strong>Date:</strong> {{ current_date }}</p>
    <hr>
    <h2>Executive Summary</h2>
    <p>{{ summary }}</p>
    <h3>Topics:</h3>
    <ul>
      {% for topic in topics %}
        <li>{{ topic }}</li>
      {% endfor %}
    </ul>

    <h2>Example Time-Series Chart</h2>
    <p style="text-align:center;">
        <img src="data:image/png;base64,{{ chart1 }}" style="max-width:600px; width:100%;">
    </p>

    <h2>Example Correlation Heatmap</h2>
    <p style="text-align:center;">
        <img src="data:image/png;base64,{{ chart2 }}" style="max-width:600px; width:100%;">
    </p>

    <footer>
        <hr>
        <small>&copy; {{ current_year }} {{ authors }}</small>
    </footer>
</body>
</html>
"""
        rendered_html = Template(template_str).render(context)

        output_html = "/tmp/report.html"
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(rendered_html)

        print(f"HTML report generated: {output_html}")

        # from weasyprint import HTML
        # pdf_path = "/tmp/report.pdf"
        # HTML(string=rendered_html).write_pdf(pdf_path)
        # print(f"PDF generated: {pdf_path}")
        # pdf_artifact = Artifact.upload_file(filepath=pdf_path, name="Report PDF", created_by_resource_name=self.__class__.__name__, bucket_name="Reports")
        html_artifact = Artifact.upload_file(filepath=output_html, name="Report HTML", created_by_resource_name=self.__class__.__name__, bucket_name="Reports")
        return html_artifact

if __name__ == "__main__":
    # Example usage:
    config = ReportConfig()  # Or override fields as needed
    app = ReportApp(config)
    html_artifact = app.run()  # Creates output_report.html and weasy_output_report.pdf
