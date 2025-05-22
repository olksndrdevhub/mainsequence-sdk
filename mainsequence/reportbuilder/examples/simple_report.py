from typing import Any, Optional, List, Union

import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

from pandas.io.formats.style import Styler
from pydantic import BaseModel

from mainsequence.reportbuilder.model import (
    Presentation,
    Theme,
    Slide,
    AbsoluteLayout,
    GridLayout,
    GridCell,
    TextElement,
    ImageElement,
    HtmlElement,
    FontWeight,
    HorizontalAlign,
    VerticalAlign,
    Position,
    Size,StyleSettings
)
from mainsequence.reportbuilder.slide_templates import _transpose_for_plotly, generic_plotly_table
from mainsequence.client.models_vam import Asset,Account
from mainsequence.client.models_helpers import MarketsTimeSeriesDetails
section_title_font_size = 11
main_title_font_size = 22
main_sequence_blue = "#003049"
title_column_width = "150px"


styles = StyleSettings()
fixed_income_usd_headers = ["INSTRUMENT", "UNITS", "PRICE", "AMOUNT", "% TOTAL", "DURATION", "YIELD", "DxV"]
fixed_income_usd_rows = [
    ["Global Note A", "16,000", "$2,900.00", "$46,400,000.00", "10.00%", "8.50", "4.30%", "3,100"],
    ["International Bond X", "40,000", "$2,200.00", "$88,000,000.00", "20.00%", "1.90", "3.90%", "700"],
    ["TOTAL", "", "", "$134,400,000.00", "30.00%", "4.10", "4.00%", "1,500"]
]

liquidity_headers = ["INSTRUMENT", "UNITS", "PRICE", "AMOUNT", "% TOTAL", "DURATION", "YIELD", "DxV"]
liquidity_rows = [
    ["Repo Agreement", "", "", "$55,000,000.00", "12.50%", "0.01", "9.50%", "5"],
    ["Cash Equiv. (Local)", "", "", "$150.00", "0.00%", "", "", ""],
    ["Cash Equiv. (USD)", "50,000", "", "$1,000,000.00", "0.20%", "", "", ""],
    ["TOTAL", "", "", "$56,000,150.00", "12.70%", "", "", ""]
]


def create_portfolio_detail_slide(target_account,fixed_income_market_ts) -> Slide:
    shared_column_widths = [1.8, 1, 1, 1.5, 0.7, 0.8, 0.8, 0.7]
    shared_cell_align = ['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']

    table_figure_width = 900


    ### Wallet
    account_latest_holdings = target_account.latest_holdings.holdings
    assets_in_holdings = Asset.filter(id__in=[a["asset"] for a in account_latest_holdings])
    assets_in_holdings_map = {a.id: a for a in assets_in_holdings}

    markets_ts = MarketsTimeSeriesDetails.get(unique_identifier=fixed_income_market_ts)
    last_risk_observation = markets_ts.related_local_time_serie.remote_table.sourcetableconfiguration.last_observation

    def get_from_last_observation(key, a_uid):

        last_value = last_risk_observation.get(a_uid, None)
        if last_value is None:
            last_value = "None"
        else:
            last_value = last_risk_observation[a_uid][key]
        return last_value

    assets = [{"ISIN": assets_in_holdings_map[h["asset"]].isin,
               "Name": assets_in_holdings_map[h["asset"]].name,
               "Price": float(h["price"]),
               "Units": float(h["quantity"]),
               "Notional":float(h["price"])*float(h["quantity"]),
               "Duration": get_from_last_observation(key="duration",
                                                   a_uid=assets_in_holdings_map[h["asset"]].unique_identifier),
               "Yield": get_from_last_observation(key="yield",
                                                     a_uid=assets_in_holdings_map[h["asset"]].unique_identifier),
               "DxV": get_from_last_observation(key="DxV",
                                                     a_uid=assets_in_holdings_map[h["asset"]].unique_identifier)

               } for h in account_latest_holdings]

    assets_df=pd.DataFrame(assets)
    assets_df["Percentage"] = assets_df["Notional"] / assets_df["Notional"].sum()

    total_row = {col: "" for col in assets_df.columns}

    # 4. Fill in the TOTAL values
    total_row["Name"] = "TOTAL"  # or whatever column you want to hold “TOTAL”
    total_row["Notional"] = assets_df["Notional"].sum()
    total_row["Percentage"] = assets_df["Percentage"].sum()

    # 5. Append it
    assets_df = pd.concat(
        [assets_df, pd.DataFrame([total_row])],
        ignore_index=True
    )

    formats = ['', '', '$,.2f', '', '', '', '', '', '.2%']

    fi_local_html = generic_plotly_table(


    headers =    assets_df.columns.to_list(),
    rows = assets_df.to_numpy().tolist(),
    column_widths = [0.7, 0.3, 0.3, 0.3],
    column_formats=formats,

    cell_align = ['left', 'right', 'right', 'right'],
    header_font_dict = dict(color='white', size=10, family=styles.chart_font_family),
    cell_font_dict = dict(size=styles.chart_label_font_size, family=styles.chart_font_family),
    fig_width = 700

    )


    fi_usd_html = generic_plotly_table(
        headers=fixed_income_usd_headers,
        rows=fixed_income_usd_rows,
        column_widths=shared_column_widths,
        cell_align=shared_cell_align,
        fig_width=table_figure_width,
        header_font_dict=dict(color='white', size=10, family=styles.chart_font_family),
        cell_font_dict=dict(size=styles.chart_label_font_size, family=styles.chart_font_family)
    )
    liquidity_html = generic_plotly_table(
        headers=liquidity_headers,
        rows=liquidity_rows,
        column_widths=shared_column_widths,
        cell_align=shared_cell_align,
        fig_width=table_figure_width,
        header_font_dict=dict(color='white', size=10, family=styles.chart_font_family),
        cell_font_dict=dict(size=styles.chart_label_font_size, family=styles.chart_font_family)
    )

    total_general_rows = [["TOTAL", "", "", "$441,687,650.00", "100.00%", "", "", ""]]

    total_table_html = generic_plotly_table(
        headers=liquidity_headers,  # Using liquidity_headers for structure, but they are hidden
        rows=total_general_rows,
        fig_width=table_figure_width,
        column_widths=shared_column_widths,
        cell_align=['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right'],  # Match length
        header_height=0,
        cell_fill_color='rgba(0,0,0,0)',
        line_color='rgba(0,0,0,0)',
        margin_dict=dict(l=0, r=0, t=5, b=0),
        cell_font_dict=dict(size=12, family=styles.chart_font_family)
    )

    slide_layout = GridLayout(
        row_definitions=["auto", "auto", "auto", "auto", "auto"],
        col_definitions=[styles.title_column_width, "1fr"],
        gap=0,
        cells=[
            GridCell(row=1, col=1, col_span=2, element=TextElement(
                text="Portfolio", font_size=styles.main_title_font_size, font_weight=FontWeight.bold,
                h_align=HorizontalAlign.left, color=styles.main_color
            ), padding="0 0 3px 0"),

            GridCell(row=2, col=1, element=TextElement(
                text="Fixed Income (Local Currency)", font_weight=FontWeight.bold,
                font_size=styles.section_title_font_size,
                color=styles.main_color, h_align=HorizontalAlign.left, v_align=VerticalAlign.center
            ), padding="2px 10px 2px 0", align_self="start"),
            GridCell(row=2, col=2, element=HtmlElement(html=fi_local_html), padding="2px 0 2px 0"),

            GridCell(row=3, col=1, element=TextElement(
                text="Fixed Income (USD)", font_weight=FontWeight.bold, font_size=styles.section_title_font_size,
                color=styles.main_color, h_align=HorizontalAlign.left, v_align=VerticalAlign.center
            ), padding="2px 10px 2px 0", align_self="start"),
            GridCell(row=3, col=2, element=HtmlElement(html=fi_usd_html), padding="2px 0 2px 0"),

            GridCell(row=4, col=1, element=TextElement(
                text="Liquidity", font_weight=FontWeight.bold, font_size=styles.section_title_font_size,
                color=styles.main_color, h_align=HorizontalAlign.left, v_align=VerticalAlign.center
            ), padding="2px 10px 2px 0", align_self="start"),
            GridCell(row=4, col=2, element=HtmlElement(html=liquidity_html), padding="2px 0 2px 0"),

            GridCell(row=5, col=2, element=HtmlElement(html=total_table_html), padding="2px 0 2px 0"),
        ]
    )

    return Slide(
        title="Portfolio Detail",
        layout=slide_layout,
        background_color=styles.default_background_color
    )


def create_full_presentation(account_uuid:str,fixed_income_market_ts:str,) -> Presentation:
    target_account=Account.get(uuid=account_uuid)

    ms_theme = Theme(
        logo_url="https://cdn.prod.website-files.com/67d166ea95c73519badbdabd/67d166ea95c73519badbdc60_Asset%25202%25404x-8-p-800.png",
        current_date="May 2025",
        font_family="Lato, Arial, Helvetica, sans-serif",
        base_font_size=11,
        title_color="#005f73"
    )
    slide1_elements = [
        TextElement(text="Monthly Report", font_size=52, font_weight=FontWeight.bold, color="#FFFFFF",
                    h_align=HorizontalAlign.left, position=Position(top="20%", left="8%")),
        TextElement(text="May 2025", font_size=32, color="#FFFFFF", h_align=HorizontalAlign.left,
                    position=Position(top="35%", left="8%")),
        ImageElement(src=ms_theme.logo_url, alt="Main Sequence Logo", size=Size(height="40px", width="auto"),
                     position=Position(top="50%", left="8%")),
        TextElement(text="Data Insights & Solutions", font_size=18, color="#FFFFFF", h_align=HorizontalAlign.left,
                    position=Position(top="calc(50% + 50px)", left="8%"))
    ]
    slide1 = Slide(title="Main Cover", background_color="#001f3f",
                   layout=AbsoluteLayout(elements=slide1_elements, width="100%", height="100%"))
    slide2 = create_portfolio_detail_slide(target_account,fixed_income_market_ts)

    presentation = Presentation(
        title="Main Sequence Monthly Report (Example)",
        subtitle="May 2025",
        theme=ms_theme,
        slides=[slide1, slide2]
    )
    return presentation


if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_html_path = output_dir / "slides_report.html"
    final_presentation = create_full_presentation(account_uuid="15b323f6-5918-4ee2-8faa-2a590dff467f",
                          fixed_income_market_ts="alpaca_1d_bars",)
    try:
        html_content = final_presentation.render()
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Presentation rendered successfully to {output_html_path}")
    except Exception as e:
        print(f"An error occurred during rendering: {e}")
        import traceback

        traceback.print_exc()