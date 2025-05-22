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
    Size
)
from mainsequence.reportbuilder.slide_templates import _transpose_for_plotly, create_plotly_table_html

section_title_font_size = 11
main_title_font_size = 22
main_sequence_blue = "#003049"
title_column_width = "150px"

fixed_income_local_headers = ["INSTRUMENT", "UNITS", "PRICE", "AMOUNT", "% TOTAL", "DURATION", "YIELD", "DxV"]
fixed_income_local_rows = [
    ["Alpha Bond 2025", "350,000", "$99.50", "$34,825,000.00", "7.50%", "0.25", "9.05%", "90"],
    ["Beta Note 2026", "160,000", "$99.80", "$15,968,000.00", "3.60%", "1.30", "9.15%", "530"],
    ["Gamma Security 2026", "250,000", "$99.90", "$24,975,000.00", "5.60%", "1.50", "9.20%", "600"],
    ["Delta Issue 2027", "245,000", "$100.10", "$24,524,500.00", "5.40%", "1.60", "9.25%", "630"],
    ["Epsilon Paper 2026", "200,000", "$98.50", "$19,700,000.00", "4.40%", "0.80", "8.30%", "300"],
    ["Zeta Bond 2029", "170,000", "$102.50", "$17,425,000.00", "3.90%", "3.30", "8.60%", "1,500"],
    ["Eta Security 2030", "180,000", "$100.00", "$18,000,000.00", "4.00%", "3.80", "8.80%", "1,700"],
    ["Theta Note 2034", "110,000", "$93.00", "$10,230,000.00", "2.30%", "6.30", "9.30%", "3,500"],
    ["Iota UDI 2028", "40,000", "$98.00", "$33,600,000.00", "7.90%", "3.20", "4.90%", "1,300"],
    ["Kappa C-Bill 2026A", "2,500,000", "$9.20", "$23,000,000.00", "5.10%", "0.85", "8.40%", "340"],
    ["Lambda C-Bill 2026B", "3,300,000", "$8.80", "$29,040,000.00", "6.70%", "1.25", "8.50%", "520"],
    ["TOTAL", "", "", "$251,287,500.00", "56.70%", "1.60", "8.55%", "480"]
]

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


def create_portfolio_detail_slide() -> Slide:
    shared_column_widths = [1.8, 1, 1, 1.5, 0.7, 0.8, 0.8, 0.7]
    shared_cell_align = ['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']

    table_figure_width = 900

    fi_local_html = create_plotly_table_html(
        fixed_income_local_headers,
        fixed_income_local_rows,
        265,
        shared_column_widths,
        shared_cell_align,
        fig_width=table_figure_width
    )
    fi_usd_html = create_plotly_table_html(
        fixed_income_usd_headers,
        fixed_income_usd_rows,
        105,
        shared_column_widths,
        shared_cell_align,
        fig_width=table_figure_width
    )
    liquidity_html = create_plotly_table_html(
        liquidity_headers,
        liquidity_rows,
        115,
        shared_column_widths,
        shared_cell_align,
        fig_width=table_figure_width
    )

    total_general_headers = ["", "", ""]
    total_general_rows = [["<b>GRAND TOTAL:</b>", "$441,687,650.00", "100.00%"]]
    total_table_height = 40
    total_table_width = 550
    cell_font_size = 9

    total_table_cell_font_style = dict(size=cell_font_size + 1, color=main_sequence_blue, family="Lato, Arial, Helvetica, sans-serif")
    total_cell_align = ['left', 'right', 'right']
    total_column_widths = [0.4, 0.3, 0.3]

    total_fig = go.Figure(data=[go.Table(
        header=dict(values=total_general_headers, height=0, line_color='rgba(0,0,0,0)'),
        cells=dict(
            values=_transpose_for_plotly(total_general_rows, len(total_general_headers)),
            fill_color='rgba(0,0,0,0)',
            font=total_table_cell_font_style,
            align=total_cell_align,
            line_color='rgba(0,0,0,0)',
            height=25
        ),
        columnwidth=total_column_widths
    )])
    total_fig.update_layout(
        height=total_table_height,
        width=total_table_width,
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    raw_total_table_html = total_fig.to_html(include_plotlyjs=False, full_html=False,
                                             config={'responsive': False, 'displayModeBar': False})
    centered_total_table_html = f'<div style="display: flex; justify-content: center; width: 100%;">{raw_total_table_html}</div>'

    slide_layout = GridLayout(
        row_definitions=["auto", "auto", "auto", "auto", "auto"],
        col_definitions=[title_column_width, "1fr"],
        gap=0,
        cells=[
            GridCell(row=1, col=1, col_span=2, element=TextElement(
                text="Portfolio", font_size=main_title_font_size, font_weight=FontWeight.bold,
                h_align=HorizontalAlign.left, color=main_sequence_blue
            ), padding="0 0 3px 0"),

            GridCell(row=2, col=1, element=TextElement(
                text="Fixed Income (Local Currency)", font_weight=FontWeight.bold, font_size=section_title_font_size,
                color=main_sequence_blue, h_align=HorizontalAlign.left, v_align=VerticalAlign.middle
            ), padding="2px 10px 2px 0", align_self="start"),
            GridCell(row=2, col=2, element=HtmlElement(html=fi_local_html), padding="2px 0 2px 0"),

            GridCell(row=3, col=1, element=TextElement(
                text="Fixed Income (USD)", font_weight=FontWeight.bold, font_size=section_title_font_size,
                color=main_sequence_blue, h_align=HorizontalAlign.left, v_align=VerticalAlign.middle
            ), padding="2px 10px 2px 0", align_self="start"),
            GridCell(row=3, col=2, element=HtmlElement(html=fi_usd_html), padding="2px 0 2px 0"),

            GridCell(row=4, col=1, element=TextElement(
                text="Liquidity", font_weight=FontWeight.bold, font_size=section_title_font_size,
                color=main_sequence_blue, h_align=HorizontalAlign.left, v_align=VerticalAlign.middle
            ), padding="2px 10px 2px 0", align_self="start"),
            GridCell(row=4, col=2, element=HtmlElement(html=liquidity_html), padding="2px 0 2px 0"),

            GridCell(row=5, col=1, col_span=2, element=HtmlElement(html=centered_total_table_html), padding="5px 0 0 0")
        ]
    )

    return Slide(
        title="Portfolio Detail",
        layout=slide_layout,
        background_color="#FFFFFF"
    )


def pie_chart_bars_slide() -> Slide:
    main_blue_color = "#003049"
    text_color_dark = "#333333"
    pie_chart_slice_colors = ['#B2DFDB', main_blue_color, '#4A90E2'] # Pale Aqua, MSBlue, Medium Blue
    bar_chart_bar_color = main_blue_color

    section_title_font_size = 16
    table_font_size_int = 10  # Integer for Plotly font size
    chart_label_font_size = 9
    chart_font_family = "Lato, Arial, Helvetica, sans-serif"

    # Data
    asset_table_headers_raw = ["ASSET CLASS", "%"]
    asset_table_headers_bold = [f"<b>{h}</b>" for h in asset_table_headers_raw]
    asset_table_cols_data = [
        ["Domestic Equities", "International Bonds", "Cash & Equivalents"],
        ["57%", "30%", "13%"]
    ]

    asset_pie_labels = ["Domestic Equities", "Cash & Equivalents", "International Bonds"]
    asset_pie_values = [57, 13, 30]

    duration_bar_labels = [">7 Years", "3-7 Years", "2-3 Years", "1-2 Years", "0-1 Years", "Short-Term/Cash"]
    duration_bar_values = [10.05, 12.07, 0.20, 41.52, 12.16, 12.76]

    # --- Element Definitions ---

    # Asset Allocation Title
    asset_allocation_title_el = TextElement(
        text=f"<span style='padding-bottom: 1px;'>Asset Allocation</span>",
        font_size=section_title_font_size, font_weight=FontWeight.bold,
        color=main_blue_color, h_align=HorizontalAlign.left
    )

    # Asset Class Table
    fig_asset_table = go.Figure(data=[go.Table(
        header=dict(
            values=asset_table_headers_bold,  # Use bolded headers
            font=dict(size=table_font_size_int, color=main_blue_color, family=chart_font_family),
            align=['left', 'right'],
            line_color=main_blue_color,  # Line for header bottom (will box header cells)
            fill_color='rgba(0,0,0,0)',
            height=28  # Adjusted height
        ),
        cells=dict(
            values=asset_table_cols_data,
            font=dict(size=table_font_size_int, color=text_color_dark, family=chart_font_family),
            align=['left', 'right'],
            fill_color='rgba(0,0,0,0)',
            line_color='lightgrey',  # Faint lines for data cells, or use main_blue_color for consistency
            height=24  # Adjusted height
        ),
        columnwidth=[0.3, 0.2]
    )])
    fig_asset_table.update_layout(
        height=125,  # Adjusted based on 3 rows + header and cell heights
        width=300,  # Set fixed width to make table thinner
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='rgba(0,0,0,0)'  # Ensure Plotly table bg is transparent
    )
    asset_class_table_html = fig_asset_table.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False})
    asset_class_table_el = HtmlElement(html=asset_class_table_html)

    # Pie Chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=asset_pie_labels, values=asset_pie_values, marker_colors=pie_chart_slice_colors,
        textinfo='percent', textfont_size=11,
        hoverinfo='label+percent+value', sort=False, showlegend=True
    )])
    fig_pie.update_layout(
        height=400, width=430,
        margin=dict(l=5, r=5, t=5, b=70),  # Increased bottom margin for legend space
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=chart_font_family),
        legend=dict(
            orientation="h",
            yanchor="top",  # Anchor legend by its top
            y=-0.2,  # Position top of legend below plot area (-0.1 to -0.3 usually works)
            xanchor="center",
            x=0.5,
            font=dict(size=chart_label_font_size, color=text_color_dark),
            bgcolor='rgba(0,0,0,0)',
            traceorder='normal'
        )
    )
    asset_pie_el = HtmlElement(html=fig_pie.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'responsive': False}))

    # Duration Breakdown Title
    duration_breakdown_title_el = TextElement(
        text="Duration Breakdown",
        font_size=section_title_font_size, font_weight=FontWeight.bold,
        color=main_blue_color, h_align=HorizontalAlign.left
    )

    # Bar Chart
    fig_bar = go.Figure(data=[go.Bar(
        y=duration_bar_labels, x=duration_bar_values, orientation='h', marker_color=bar_chart_bar_color,
        text=[f"{val:.2f}%" for val in duration_bar_values],
        textposition='outside', textfont_size=chart_label_font_size, hoverinfo='y+x',
    )])
    max_val = max(duration_bar_values)
    fig_bar.update_layout(
        # height is removed to allow responsive fill, assuming container gives height
        margin=dict(l=20, r=20, t=5, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False, zeroline=False, showline=False,
            showticklabels=True, ticksuffix='%', side='bottom',
            tickfont=dict(size=chart_label_font_size, color=text_color_dark),
            range=[0, max_val * 1.115]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showline=False,
            autorange="reversed",
            tickfont=dict(size=chart_label_font_size, color=text_color_dark)
        ),
        bargap=0.4,
        font=dict(family=chart_font_family),
        width=450,
    )
    duration_bar_el = HtmlElement(html=fig_bar.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'responsive': True}))

    # --- Main Slide Layout ---
    slide_layout = GridLayout(
        row_definitions=["auto", "auto", "1fr"],
        col_definitions=["1fr", "1fr"],
        gap=10,
        cells=[
            # Column 1 (Left)
            GridCell(row=1, col=1, element=asset_allocation_title_el,
                     align_self=VerticalAlign.bottom, padding="0 0 2px 0"),
            GridCell(row=2, col=1, element=asset_class_table_el,
                     align_self=VerticalAlign.top,
                     justify_self=HorizontalAlign.center,
                     padding="5px 0 0 0"),
            GridCell(row=3, col=1, element=asset_pie_el,
                     align_self=VerticalAlign.middle, justify_self=HorizontalAlign.center, padding="5px 0 0 0"),

            # Column 2 (Right)
            GridCell(row=1, col=2, element=duration_breakdown_title_el,
                     align_self=VerticalAlign.bottom, padding="0 0 2px 0"),
            GridCell(row=2, col=2, row_span=2, element=duration_bar_el,  # Bar chart spans row 2 and 3
                     align_self=VerticalAlign.middle,  # Stretch the element within the cell
                     justify_self=HorizontalAlign.left,  # Default, content will stretch if bar_el is 100% width
                     padding="5px 0 0 0")
        ]
    )

    return Slide(
        title="Portfolio Snapshot",
        layout=slide_layout,
        background_color="#FFFFFF"
    )

def create_full_presentation() -> Presentation:
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

    presentation = Presentation(
        title="Main Sequence Monthly Report (Example)",
        subtitle="May 2025",
        theme=ms_theme,
        slides=[
            Slide(title="Main Cover", background_color="#001f3f",
                  layout=AbsoluteLayout(elements=slide1_elements, width="100%", height="100%")),
            create_portfolio_detail_slide(),
            pie_chart_bars_slide()
        ]
    )
    return presentation




if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_html_path = output_dir / "ms_template_report.html"
    final_presentation = create_full_presentation()
    try:
        html_content = final_presentation.render()
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Presentation rendered successfully to {output_html_path}")
    except Exception as e:
        print(f"An error occurred during rendering: {e}")
        import traceback

        traceback.print_exc()