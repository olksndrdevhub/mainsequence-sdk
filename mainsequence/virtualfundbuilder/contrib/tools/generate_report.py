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
from weasyprint import HTML

from mainsequence.virtualfundbuilder.models import ReportConfig
from mainsequence.virtualfundbuilder.resource_factory.tool_factory import register_tool, BaseTool


@register_tool()
class ReportTool(BaseTool):
    """
    Minimal example of a 'ReportTool' that can:
    1) Generate dummy data and create charts (line + heatmap).
    2) Embed those charts into an HTML template.
    3) Optionally export the HTML to PDF using WeasyPrint.
    """
    configuration = ReportConfig

    def run(self, config: ReportConfig):
        print("running tool")

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

    def run(self, config: ReportConfig, include_pdf: bool = True):
        """
        Generates an HTML report (and optional PDF) in a minimal, self-contained way.
        """

        # Create base64-encoded charts
        chart1, chart2 = self._generate_charts()

        # Build context from config
        context = {
            "current_date": datetime.now().strftime('%Y-%m-%d'),
            "current_year": datetime.now().year,
            "chart1": chart1,
            "chart2": chart2,
            # Pulling fields from our pydantic config:
            "report_id": self.config.report_id,
            "authors": self.config.authors,
            "sector": self.config.sector,
            "region": self.config.region,
            "topics": self.config.topics,
            "report_title": self.config.report_title,
            "summary": self.config.summary,
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

        # Render the template:
        rendered_html = Template(template_str).render(context)

        # Write HTML to file
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(rendered_html)

        print(f"HTML report generated: {output_html}")

        # Optionally convert to PDF with WeasyPrint
        if include_pdf and self.config.pdf_path:
            HTML(string=rendered_html).write_pdf(self.config.pdf_path)
            print(f"PDF generated: {self.config.pdf_path}")


if __name__ == "__main__":
    # Example usage:
    config = ReportConfig()  # Or override fields as needed
    tool = ReportTool(config)
    tool.generate_report()  # Creates output_report.html and weasy_output_report.pdf
