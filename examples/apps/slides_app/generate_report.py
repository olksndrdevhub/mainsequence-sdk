


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

from plotly.subplots import make_subplots

from mainsequence.client import DoesNotExist, AssetCategory,Asset, Account
from mainsequence.client.models_tdag import Artifact
from mainsequence.virtualfundbuilder.resource_factory.app_factory import register_app, BaseApp
#!/usr/bin/env python3



# ----- Utility functions for DRY HTML construction -----
def column_wrapper(content: str) -> str:
    """Wraps given HTML content inside a flex column div."""
    return (
        '<div class="col-6 d-flex flex-column h-100 p-0" style="min-height:0;">'
        f'{content}'
        '</div>'
    )


def two_column_layout(left_content: str, right_content: str) -> str:
    """Creates a two-column row layout with provided left and right column contents."""
    return (
        '<div class="row g-0 h-100">'
        f'{column_wrapper(left_content)}'
        f'{column_wrapper(right_content)}'
        '</div>'
    )


def padded_div(content: str, pt: int = 0, pb: int = 0, pl: int = 0, pr: int = 0, flex_grow: bool = False) -> str:
    """Creates a div with specified padding. Optionally adds flex-grow classes."""
    classes = "flex-grow-1 position-relative" if flex_grow else ""
    style = f"padding:{pt}px {pr}px {pb}px {pl}px; min-height:0;"
    return f'<div class="{classes}" style="{style}">{content}</div>'


class ReportConfig(BaseModel):
    """Pydantic model defining the parameters for report generation."""
    presentation_title: str = "Presentation"
    presentation_subtitle: str = "Presentation Subtitle"
    account_uuid: str
    fixed_income_market_ts:str
    logo_url:str
    current_date:str


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

        target_account = Account.get(uuid=self.configuration.account_uuid)
        self.target_account = target_account

    def _build_interactive_chart_slide(self) -> str:
        # ----- Constants -----
        palette = ['#0A1E3C', '#00A3E0', '#4A5A7A']
        PIE_HEIGHT_PX = 350
        BAR_ROW_PX = 40
        BAR_PADDING_PX = 60
        MAX_BAR_HEIGHT = 350

        # ----- Data -----
        pie_vals, pie_labels = [30, 45, 25], ['A', 'B', 'C']
        cats = [f'Category {i}' for i in range(1, 11)]
        vals = [i * 5 for i in range(1, 11)]

        # ----- Pie + Table stacked vertically -----
        fig = make_subplots(
            rows=2, cols=1,
            specs=[
                [{"type": "domain"}],
                [{"type": "table"}]
            ],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.02
        )

        # pie trace with labels+percent, legend off
        fig.add_trace(
            go.Pie(
                labels=pie_labels,
                values=pie_vals,
                hole=0.3,
                marker=dict(colors=palette),
                textinfo='label+percent',  # show labels on slices
                showlegend=False  # hide the legend
            ),
            row=1, col=1
        )

        # table trace below
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Label", "Value"],
                    font=dict(size=14, color="white"),
                    fill_color="grey",
                    align="left"
                ),
                cells=dict(
                    values=[pie_labels, pie_vals],
                    font=dict(size=12),
                    fill_color="white",
                    align="left"
                )
            ),
            row=2, col=1
        )

        # transparent background and sizing
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=PIE_HEIGHT_PX,
            margin=dict(l=0, r=0, t=20, b=0)
        )

        combined_html = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            div_id='pie-table-chart-int',
            default_width='100%',
            default_height=f'{PIE_HEIGHT_PX}px',
            config={'responsive': True}
        )
        left_col = padded_div(combined_html, flex_grow=True)

        # ----- Bar Chart HTML -----
        natural_height = BAR_ROW_PX * len(cats) + BAR_PADDING_PX
        bar_height_px = min(natural_height, MAX_BAR_HEIGHT)
        bar_fig = px.bar(x=vals, y=cats, orientation='h',
                         color_discrete_sequence=palette * 4)
        bar_fig.update_layout(
            height=bar_height_px,
            xaxis_title='Value',
            yaxis_title=None,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

        )
        bar_html = bar_fig.to_html(
            full_html=False, include_plotlyjs=False,
            div_id='bar-chart-int',
            default_width='100%', default_height=f'{bar_height_px}px',
            config={'responsive': True}
        )
        bar_html = padded_div(bar_html, flex_grow=True)

        # ----- Assemble two-column layout -----
        return two_column_layout(left_col, bar_html)

    def _build_slide_table_html(self):
        # 2) Define 3 example assets

        account_latest_holdings=self.target_account.latest_holdings.holdings
        assets_in_holdings=Asset.filter(id__in=[a["asset"] for a in account_latest_holdings])
        assets_in_holdings_map={a.id:a for a in assets_in_holdings}

        #get asset risk measurements
        from mainsequence.client.models_helpers import MarketsTimeSeriesDetails
        from mainsequence.tdag import APITimeSerie
        markets_ts=MarketsTimeSeriesDetails.get(unique_identifier=self.configuration.fixed_income_market_ts)

        last_risk_observation=markets_ts.related_local_time_serie.remote_table.sourcetableconfiguration.last_observation

        def get_from_last_observation(key,a_uid):

            last_value=last_risk_observation.get(a_uid,None)
            if last_value is None:
                last_value= "None"
            else:
                last_value=last_risk_observation[a_uid][key]
            return last_value

        assets= [{"isin":assets_in_holdings_map[h["asset"]].isin,
                  "name": assets_in_holdings_map[h["asset"]].name,
                  "price":float(h["price"]),
                  "units":float(h["quantity"]),

                  "volume":get_from_last_observation(key="volume",a_uid=assets_in_holdings_map[h["asset"]].unique_identifier)

                  } for h in account_latest_holdings]

        # 3) Compute notional and total
        for a in assets:
            a["notional"] = a["units"] * a["price"]
        total_notional = sum(a["notional"] for a in assets)

        # 4) Compute percentage share
        for a in assets:
            a["percent"] = round((a["notional"] / total_notional) * 100, 2)

        # 5) Build the HTML table string
        table_rows = "\n".join(
            f"<tr>"
            f"<td>{a['isin']}</td>"
            f"<td>{a['name']}</td>"
            f"<td>{a['units']}</td>"
            f"<td>{a['price']:.2f}</td>"
            f"<td>{a['notional']:.2f}</td>"
            f"<td>{a['percent']:.2f}%</td>"
             f"<td>{a['volume']:.2f}%</td>"
            f"</tr>"
            for a in assets
        )

        table_html = f"""
        <table class="table table-sm table-borderless bg-transparent">
          <thead style="background-color: #cce5ff;">
            <tr>
              <th>ISIN</th>
               <th>Name</th>
              <th>Units</th>
              <th>Price</th>
              <th>Notional</th>
              <th>Volume</th>
              <th>%</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
          <tfoot>
            <tr>
              <th colspan="3">Total</th>
              <th>{total_notional:.2f}</th>
              <th>100.00%</th>
            </tr>
          </tfoot>
        </table>
        """

        return padded_div(content=table_html, pt=15, pb=15, pl=15, pr=15,)


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

        slides.append({
            "title": "Assets Overview",
            "content": self._build_slide_table_html()
        })


        slides.append({
            "title": "Asset Charts Interactive",
            "content": self._build_interactive_chart_slide()
        })

        # 3) Build context for Jinja2
        template_context = {
            "presentation_title": self.configuration.presentation_title,
            "presentation_subtitle": self.configuration.presentation_subtitle,
            "slides": slides,
            "logo_url":self.configuration.logo_url,
            "current_date":self.configuration.current_date,
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
    config = ReportConfig(account_uuid="15b323f6-5918-4ee2-8faa-2a590dff467f",
                          fixed_income_market_ts="alpaca_1d_bars",
                          current_date=datetime.now().date().strftime('%d-%b-%y'),
                          logo_url="https://cdn.prod.website-files.com/67d166ea95c73519badbdabd/67d166ea95c73519badbdc60_Asset%25202%25404x-8-p-800.png"
                          )  # Or override fields as needed
    app = SlideReport(config)
    html_artifact = app.run()  # Creates output_report.html and weasy_output_report.pdf
