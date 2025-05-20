


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

from mainsequence.tdag import APITimeSerie
from mainsequence.virtualfundbuilder.utils import get_vfb_logger

from pydantic import BaseModel
from jinja2 import Template


from mainsequence.client import DoesNotExist, AssetCategory,TargetPortfolio
from mainsequence.client.models_tdag import Artifact
from mainsequence.virtualfundbuilder.resource_factory.app_factory import register_app, BaseApp
#!/usr/bin/env python3


class ReportConfig(BaseModel):
    """Pydantic model defining the parameters for report generation."""
    portfolio_ticker: str


@register_app()
class SlideReport(BaseApp):
    """
    Minimal example of a 'ReportApp' that can:
    1) Generate dummy data and create charts (line + heatmap).
    2) Embed those charts into an HTML template.
    3) Optionally export the HTML to PDF using WeasyPrint.
    """
    configuration_class = ReportConfig
    def __init__(self, configuration: ReportConfig):
        self.configuration = configuration

        target_portfolio = TargetPortfolio.get(ticker=self.configuration.portfolio_ticker)
        self.target_portfolio = target_portfolio


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
        Generates an HTML presentation with two mock slides based on the Jinja2 template,
        saving to 'output_presentation.html'.
        """
        # 1) Log the configuration
        print(f"Running PresentationTool with configuration: {self.configuration}")

        # 2) Create two mock slides
        slides = [
            {
                "title": "Welcome",
                "content": "<p>Thank you for attending this demo presentation.</p>"
            },
            {
                "title": "Next Steps",
                "content": (
                    "<ul>"
                    "<li>Review the template</li>"
                    "<li>Customize your own slides</li>"
                    "<li>Export to PDF using the button</li>"
                    "</ul>"
                )
            }
        ]

        # 3) Build context for Jinja2
        template_context = {
            "title": self.configuration.get("title", "Demo Presentation"),
            "slides": slides
        }

        # 4) Setup Jinja2 environment (point to templates directory)
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)
        template = env.get_template('presentation_template.html')

        # 5) Render the HTML
        rendered_html = template.render(template_context)

        # 6) Write the rendered HTML to disk
        output_path = os.path.join(os.path.dirname(__file__), 'output_presentation.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)

        print(f"Generated HTML presentation at: {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage:
    config = ReportConfig()  # Or override fields as needed
    app = SlideReport(config)
    html_artifact = app.run()  # Creates output_report.html and weasy_output_report.pdf
