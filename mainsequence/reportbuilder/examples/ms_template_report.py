from typing import Any, Optional, List, Union

from pathlib import Path
import plotly.graph_objects as go
from pydantic import BaseModel # Added for ReportStyleSettings

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
    Size, StyleSettings
)
from mainsequence.reportbuilder.slide_templates import (
    generic_plotly_table,
    generic_plotly_pie_chart,
    generic_plotly_bar_chart
)

# Instantiate the style settings
styles = StyleSettings()


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
    shared_cell_align: Union[str, List[str]] = ['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
    table_figure_width = 900

    fi_local_html = generic_plotly_table(
        headers=fixed_income_local_headers,
        rows=fixed_income_local_rows,
        column_widths=shared_column_widths,
        cell_align=shared_cell_align,
        fig_width=table_figure_width,
        header_font_dict=dict(color='white', size=10, family=styles.chart_font_family),
        cell_font_dict=dict(size=styles.chart_label_font_size, family=styles.chart_font_family)
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
        headers=liquidity_headers, # Using liquidity_headers for structure, but they are hidden
        rows=total_general_rows,
        fig_width=table_figure_width,
        column_widths=shared_column_widths,
        cell_align=['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right'], # Match length
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
                text="Fixed Income (Local Currency)", font_weight=FontWeight.bold, font_size=styles.section_title_font_size,
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


def pie_chart_bars_slide() -> Slide:
    pie_chart_slice_colors = ['#B2DFDB', styles.main_color, '#4A90E2']

    asset_table_headers_raw = ["ASSET CLASS", "%"]
    asset_table_rows_data = [
        ["Domestic Equities", "57%"],
        ["International Bonds", "30%"],
        ["Cash & Equivalents", "13%"]
    ]

    asset_class_table_html = generic_plotly_table(
        headers=asset_table_headers_raw,
        rows=asset_table_rows_data,
        table_height=125,
        fig_width=300,
        column_widths=[0.3, 0.2],
        header_font_dict=dict(color='white', size=10, family=styles.chart_font_family),
        cell_font_dict=dict(size=styles.chart_label_font_size, family=styles.chart_font_family)
    )
    asset_class_table_el = HtmlElement(html=asset_class_table_html)

    asset_pie_labels = ["Domestic Equities", "Cash & Equivalents", "International Bonds"]
    asset_pie_values = [57, 13, 30]
    asset_pie_html = generic_plotly_pie_chart(
        labels=asset_pie_labels,
        values=asset_pie_values,
        colors=pie_chart_slice_colors,
        height=400,
        width=430,
        textinfo='percent',
        legend_dict = dict(
            font=dict(size=styles.chart_label_font_size, family=styles.chart_font_family),
            orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)'
        ),
        font_dict=dict(family=styles.chart_font_family)
    )
    asset_pie_el = HtmlElement(html=asset_pie_html)

    duration_bar_labels = [">7 Years", "3-7 Years", "2-3 Years", "1-2 Years", "0-1 Years", "Short-Term/Cash"]
    duration_bar_values = [10.05, 12.07, 0.20, 41.52, 12.16, 12.76]

    max_val = max(duration_bar_values)
    custom_xaxis_bar_dict = dict(
        showgrid=False, zeroline=False, showline=False,
        showticklabels=True, ticksuffix='%', side='bottom',
        tickfont=dict(size=styles.chart_label_font_size, color=styles.text_color_dark, family=styles.chart_font_family),
        range=[0, max_val * 1.115]
    )
    custom_yaxis_bar_dict = dict(
        showgrid=False, zeroline=False, showline=False,
        tickfont=dict(size=styles.chart_label_font_size, color=styles.text_color_dark, family=styles.chart_font_family),
        autorange="reversed"
    )


    duration_bar_html = generic_plotly_bar_chart(
        y_values=duration_bar_labels,
        x_values=duration_bar_values,
        orientation='h',
        bar_color=styles.main_color,
        width=450,
        height=450,
        xaxis_dict=custom_xaxis_bar_dict,
        yaxis_dict=custom_yaxis_bar_dict,
        font_dict=dict(family=styles.chart_font_family),
        textfont_dict=dict(size=styles.chart_label_font_size, family=styles.chart_font_family)
    )
    duration_bar_el = HtmlElement(html=duration_bar_html)

    slide_layout = GridLayout(
        row_definitions=["auto", "auto", "1fr"],
        col_definitions=["1fr", "1fr"],
        gap=10,
        cells=[
            GridCell(row=1, col=1, element=TextElement(
                text=f"<span style='padding-bottom: 1px;'>Asset Allocation</span>",
                font_size=styles.pie_chart_section_title_font_size, font_weight=FontWeight.bold,
                color=styles.main_color, h_align=HorizontalAlign.left
            ), align_self=VerticalAlign.bottom, padding="0 0 2px 0"),
            GridCell(row=2, col=1, element=asset_class_table_el,
                     align_self=VerticalAlign.top, justify_self=HorizontalAlign.center, padding="5px 0 0 0"),
            GridCell(row=3, col=1, element=asset_pie_el,
                     align_self=VerticalAlign.center, justify_self=HorizontalAlign.center, padding="5px 0 0 0"),

            GridCell(row=1, col=2, element=TextElement(
                text="Duration Breakdown", font_size=styles.pie_chart_section_title_font_size, font_weight=FontWeight.bold,
                color=styles.main_color, h_align=HorizontalAlign.left
            ), align_self=VerticalAlign.bottom, padding="0 0 2px 0"),
            GridCell(row=2, col=2, row_span=2, element=duration_bar_el,
                     align_self=VerticalAlign.center, justify_self=HorizontalAlign.left, padding="5px 0 0 0")
        ]
    )

    return Slide(
        title="Portfolio Snapshot",
        layout=slide_layout,
        background_color=styles.default_background_color
    )


def create_generic_table_and_grouped_bar_chart_slide() -> Slide:
    slide_main_title_text = "Component Analysis: Multi-Series Data"

    table_headers = ["Type", "Value Set 1", "Value Set 2", "Difference (VS1-VS2)"]
    raw_table_rows_data = [
        ["Alpha Group", 0.60, 0.40],
        ["Beta Group",  0.45, 0.50],
        ["Gamma Group", 0.70, 0.55],
        ["Delta Group", 0.25, 0.25],
        ["Epsilon Group",0.50, 0.30],
        ["Zeta Group",  0.10, 0.15],
    ]

    table_rows_processed = []
    total_vs1 = 0
    total_vs2 = 0
    total_diff = 0

    for row_data in raw_table_rows_data:
        category, vs1, vs2 = row_data
        diff = round(vs1 - vs2, 2)
        table_rows_processed.append([category, f"{vs1:.2f}", f"{vs2:.2f}", f"{diff:.2f}"])
        total_vs1 += vs1
        total_vs2 += vs2
        total_diff += diff

    table_rows_processed.append([
        "<b>TOTAL</b>",
        f"<b>{total_vs1:.2f}</b>",
        f"<b>{total_vs2:.2f}</b>",
        f"<b>{total_diff:.2f}</b>"
    ])

    generic_table_html = generic_plotly_table(
        headers=table_headers,
        rows=table_rows_processed,
        column_widths=[0.7, 0.3, 0.3, 0.3],
        cell_align=['left', 'right', 'right', 'right'],
        header_font_dict=dict(color='white', size=10, family=styles.chart_font_family),
        cell_font_dict=dict(size=styles.chart_label_font_size, family=styles.chart_font_family),
        fig_width=700
    )

    chart_categories = [f"Point {i+1}" for i in range(11)]
    series_a_values = [0.50, 0.45, 0.60, 0.30, 0.75, 0.20, 0.55, 0.40, 0.65, 0.35, 0.70]
    series_b_values = [0.40, 0.50, 0.55, 0.35, 0.60, 0.25, 0.60, 0.30, 0.70, 0.25, 0.65]
    series_c_values = [round(a - b, 2) for a, b in zip(series_a_values, series_b_values)]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name='Data Series A',
        x=chart_categories,
        y=series_a_values,
        marker_color=styles.main_color
    ))
    fig_bar.add_trace(go.Bar(
        name='Data Series B',
        x=chart_categories,
        y=series_b_values,
        marker_color='rgb(200,200,200)'
    ))
    fig_bar.add_trace(go.Bar(
        name='Difference (A-B)',
        x=chart_categories,
        y=series_c_values,
        marker_color='rgb(160,120,70)'
    ))

    fig_bar.update_layout(
        width=800,
        height=380,
        barmode='group',
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=styles.chart_label_font_size, family=styles.chart_font_family),  # Adjusted: Font consistent with styles
            bgcolor='rgba(0,0,0,0)'  # Added: Ensures transparent background
        ),
        margin=dict(l=30, r=20, t=10, b=120),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgb(220,220,220)',
            zerolinecolor='rgb(180,180,180)',
            tickformat=".2f"
        ),
        xaxis=dict(
           tickfont=dict(size=9)
        ),
        font=dict(family=styles.chart_font_family)
    )
    generic_bar_chart_html = fig_bar.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False})

    slide_layout = GridLayout(
        row_definitions=["auto", "auto", "auto"],
        col_definitions=["1fr"],
        gap=10,
        cells=[
            GridCell(row=1, col=1, element=TextElement(
                text=slide_main_title_text,
                font_size=16,
                font_weight=FontWeight.bold,
                color=styles.main_color,
                h_align=HorizontalAlign.center
            ), padding="0 0 2px 0"),
            GridCell(
                row=2,
                col=1,
                element=HtmlElement(html=generic_table_html),
                justify_self=HorizontalAlign.center,
                content_h_align=HorizontalAlign.center, # For horizontal centering of the table
                content_v_align=VerticalAlign.center    # For vertical centering of the table
            ),
            GridCell(row=3, col=1, element=HtmlElement(html=generic_bar_chart_html), justify_self=HorizontalAlign.center, padding="5px 0 0 0")
        ]
    )

    return Slide(
        title="Generic Grouped Bar Chart and Table",
        layout=slide_layout,
        background_color="#FFFFFF"
    )


def create_full_presentation() -> Presentation:
    ms_theme = Theme(
        logo_url="https://cdn.prod.website-files.com/67d166ea95c73519badbdabd/67d166ea95c73519badbdc60_Asset%25202%25404x-8-p-800.png",
        current_date="May 2025",
        font_family=styles.chart_font_family, # Using chart_font_family as the base
        base_font_size=styles.section_title_font_size, # Using section_title_font_size as base
        title_color="#005f73" # This could also be part of ReportStyleSettings if needed globally
    )
    slide1_elements = [
        TextElement(text="Monthly Report", font_size=styles.cover_slide_main_title_font_size, font_weight=FontWeight.bold, color=styles.cover_slide_text_color,
                    h_align=HorizontalAlign.left, position=Position(top="20%", left=styles.cover_slide_element_left_indent)),
        TextElement(text="May 2025", font_size=styles.cover_slide_subtitle_font_size, color=styles.cover_slide_text_color, h_align=HorizontalAlign.left,
                    position=Position(top="35%", left=styles.cover_slide_element_left_indent)),
        ImageElement(src=ms_theme.logo_url, alt="Main Sequence Logo", size=Size(height=styles.cover_slide_logo_height, width="auto"),
                     position=Position(top="50%", left=styles.cover_slide_element_left_indent)),
        TextElement(text="Data Insights & Solutions", font_size=styles.cover_slide_tagline_font_size, color=styles.cover_slide_text_color, h_align=HorizontalAlign.left,
                    position=Position(top=f"calc(50% + {styles.cover_slide_logo_height} + 10px)", left=styles.cover_slide_element_left_indent)) # Adjusted top position
    ]

    presentation = Presentation(
        title="Main Sequence Monthly Report (Example)",
        subtitle="May 2025",
        theme=ms_theme,
        slides=[
            Slide(title="Main Cover", background_color=styles.cover_slide_background_color,
                  layout=AbsoluteLayout(elements=slide1_elements, width="100%", height="100%")),
            create_portfolio_detail_slide(),
            pie_chart_bars_slide(),
            create_generic_table_and_grouped_bar_chart_slide()
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