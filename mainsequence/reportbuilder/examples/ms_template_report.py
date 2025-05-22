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
    # --- Style Constants ---
    main_blue_color = "#003049"  # Similar to Actinver's blue
    text_color_dark = "#333333"
    pie_chart_slice_colors = ['#B8A978', '#808080', main_blue_color]  # Khaki, Grey, Dark Blue
    bar_chart_bar_color = main_blue_color

    section_title_font_size = 16  # Adjusted for visual hierarchy
    table_font_size = "10px"
    chart_label_font_size = 9

    # --- Left Column: Instrument Table HTML ---
    instrument_table_html = f"""
<table style="width: 95%; border-collapse: collapse; margin-top: 5px;">
  <thead>
    <tr style="border-bottom: 1.5px solid {main_blue_color};">
      <th style="text-align:left; padding: 5px 3px; font-weight:bold; color: {main_blue_color}; font-size: {table_font_size};">TIPO DE INSTRUMENTO</th>
      <th style="text-align:right; padding: 5px 3px; font-weight:bold; color: {main_blue_color}; font-size: {table_font_size};">%</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="text-align:left; padding: 5px 3px; font-size: {table_font_size}; color: {text_color_dark};">RENTA FIJA PESOS</td><td style="text-align:right; padding: 5px 3px; font-size: {table_font_size}; color: {text_color_dark};">57%</td></tr>
    <tr><td style="text-align:left; padding: 5px 3px; font-size: {table_font_size}; color: {text_color_dark};">RENTA FIJA DOLARES</td><td style="text-align:right; padding: 5px 3px; font-size: {table_font_size}; color: {text_color_dark};">30%</td></tr>
    <tr><td style="text-align:left; padding: 5px 3px; font-size: {table_font_size}; color: {text_color_dark};">LIQUIDEZ</td><td style="text-align:right; padding: 5px 3px; font-size: {table_font_size}; color: {text_color_dark};">13%</td></tr>
  </tbody>
</table>
"""

    # --- Left Column: Pie Chart ---
    pie_labels_legend = ["GUBERNAMENTAL", "RENTA FIJA DOLARES", "LIQUIDEZ"]
    pie_values = [57, 30, 13]

    fig_pie = go.Figure(data=[go.Pie(
        labels=pie_labels_legend,
        values=pie_values,
        marker_colors=pie_chart_slice_colors,
        textinfo='percent',
        textfont_size=10,
        hoverinfo='label+percent+value',
        sort=False,  # Keep order as specified
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.01,  # Adjusted for closeness
            xanchor="center", x=0.5,
            font=dict(size=chart_label_font_size, color=text_color_dark),
            bgcolor='rgba(0,0,0,0)'
        )
    )])
    fig_pie.update_layout(
        height=260,
        width=280,  # Fixed width for pie
        margin=dict(l=10, r=10, t=5, b=35),  # Bottom margin for legend
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Lato, Arial, Helvetica, sans-serif")
    )
    pie_html = fig_pie.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'responsive': False})

    # --- Right Column: Bar Chart ---
    bar_y_labels = [">7", "3-7", "2-3", "1-2", "0-1", "Efectivo"]
    bar_x_values = [10.05, 12.07, 0.00, 41.52, 12.16, 12.76]

    fig_bar = go.Figure(data=[go.Bar(
        y=bar_y_labels,
        x=bar_x_values,
        orientation='h',
        marker_color=bar_chart_bar_color,
        text=[f"{val:.2f}%" if val > 0.005 else "" for val in bar_x_values],  # Show % if value is not effectively zero
        textposition='outside',
        textfont_size=chart_label_font_size,
        hoverinfo='y+x'
    )])
    fig_bar.update_layout(
        height=280,
        margin=dict(l=40, r=35, t=0, b=25),  # l for y-axis labels, r for text values, b for x-axis
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            ticksuffix='%',
            side='bottom',
            tickfont=dict(size=chart_label_font_size, color=text_color_dark)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            autorange="reversed",  # Ensures ">7" is at the top
            tickfont=dict(size=chart_label_font_size, color=text_color_dark)
        ),
        bargap=0.35,
        font=dict(family="Lato, Arial, Helvetica, sans-serif")
    )
    bar_html = fig_bar.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False, 'responsive': True})

    # --- Define Column Layouts ---
    left_column_layout = GridLayout(
        row_definitions=["auto", "auto", "minmax(200px, auto)"],
        col_definitions=["1fr"],
        gap=8,
        cells=[
            GridCell(row=1, col=1, element=TextElement(
                text=f"<span style='border-bottom: 2px solid {main_blue_color}; padding-bottom: 1px;'>Cartera</span>",
                font_size=section_title_font_size, font_weight=FontWeight.bold,
                color=main_blue_color, h_align=HorizontalAlign.left
            )),
            GridCell(row=2, col=1, element=HtmlElement(html=instrument_table_html),
                     align_self=VerticalAlign.top),
            GridCell(row=3, col=1, element=HtmlElement(html=pie_html),
                     align_self=VerticalAlign.center, justify_self=HorizontalAlign.center)
        ]
    )

    right_column_layout = GridLayout(
        row_definitions=["auto", "1fr"],
        col_definitions=["1fr"],
        gap=8,
        cells=[
            GridCell(row=1, col=1, element=TextElement(
                text="Desglose por Duración",
                font_size=section_title_font_size, font_weight=FontWeight.bold,
                color=main_blue_color, h_align=HorizontalAlign.left
            )),
            GridCell(row=2, col=1, element=HtmlElement(html=bar_html),
                     align_self=VerticalAlign.top, justify_self=HorizontalAlign.left)
        ]
    )

    # --- Define Main Slide Layout ---
    slide_layout = GridLayout(
        row_definitions=["1fr"],
        col_definitions=["2fr", "3fr"],  # Relative widths for left and right columns
        gap=35,  # Gap between the two main columns
        cells=[
            GridCell(row=1, col=1, element=left_column_layout,
                     padding="0 10px 0 0", align_self=VerticalAlign.top),  # Add some padding to the right of left col
            GridCell(row=1, col=2, element=right_column_layout,
                     padding="0 0 0 10px", align_self=VerticalAlign.top)  # Add some padding to the left of right col
        ]
    )

    return Slide(
        title="Cartera y Duración",  # This title appears in the slide's header bar
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
    slide1 = Slide(title="Main Cover", background_color="#001f3f",
                   layout=AbsoluteLayout(elements=slide1_elements, width="100%", height="100%"))
    slide2 = create_portfolio_detail_slide()

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