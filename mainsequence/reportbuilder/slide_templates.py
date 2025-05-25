from typing import List, Dict, Any, Optional, Union
import plotly.graph_objects as go
from mainsequence.reportbuilder.model import (
    Slide,
    GridLayout, GridCell, AbsoluteLayout,
    TextElement, FunctionElement, HtmlElement, ImageElement,
    HorizontalAlign, VerticalAlign, FontWeight,
    Size, Position, ThemeMode,get_theme_settings
)


def _transpose_for_plotly(data_rows: List[List[Any]], num_columns: int) -> List[List[Any]]:
    if not data_rows:
        return [[] for _ in range(num_columns)]
    transposed = list(map(list, zip(*data_rows)))
    return transposed


def generic_plotly_table(
        headers: List[str],
        rows: List[List[Any]],
        table_height: Optional[int] = None,  # MODIFIED: Made optional for auto-sizing
        fig_width: Optional[int] = None,
        column_widths: Optional[List[Union[int, float]]] = None,
        cell_align: Union[str, List[str]] = 'left',
        header_align: str = 'center',
        cell_font_dict: Optional[Dict[str, Any]] = None,
        header_font_dict: Optional[Dict[str, Any]] = None,
        header_fill_color: str = 'rgb(0, 32, 96)',
        header_font_color: str = 'white',
        cell_fill_color: str = 'rgb(240,245,255)',
        line_color: str = 'rgb(200,200,200)',
        header_height: int = 22,
        cell_height: int = 20,
        margin_dict: Optional[Dict[str, int]] = None,
        paper_bgcolor: str = 'rgba(0,0,0,0)',
        plot_bgcolor: str = 'rgba(0,0,0,0)',
        responsive: bool = True,
        display_mode_bar: bool = False,
        include_plotlyjs: bool = False,
        full_html: bool = False,
        column_formats: Optional[List[str]] = None,

) -> str:
    effective_margin_dict = margin_dict if margin_dict is not None else dict(l=5, r=5, t=2, b=2)

    if cell_font_dict is None:
        cell_font_dict = dict(size=9)
    if header_font_dict is None:
        header_font_dict = dict(color=header_font_color, size=10)

    plotly_column_data = _transpose_for_plotly(rows, len(headers))
    # Build cell properties, injecting formats if provided
    cell_props = dict(
        values=plotly_column_data,
        fill_color=cell_fill_color,
        font=cell_font_dict,
        align=cell_align,
        line_color=line_color,
        height=cell_height
    )
    if column_formats:
        cell_props['format'] = column_formats


    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color=header_fill_color,
            font=header_font_dict,
            align=header_align,
            line_color=line_color,
            height=header_height if headers else 0
        ),
        cells=cell_props,
        columnwidth=column_widths if column_widths else []
    )])

    determined_fig_height: int
    if table_height is None:
        content_actual_height = (header_height if headers else 0) + (len(rows) * cell_height)
        # Figure height needs to include its own top/bottom margins
        determined_fig_height = content_actual_height + \
                                effective_margin_dict.get('t', 0) + \
                                effective_margin_dict.get('b', 0) + \
                                4  # Small buffer for any internal Plotly paddings
    else:
        determined_fig_height = table_height

    layout_args = {
        "height": determined_fig_height,
        "margin": effective_margin_dict,
        "paper_bgcolor": paper_bgcolor,
        "plot_bgcolor": plot_bgcolor
    }
    if fig_width:
        layout_args["width"] = fig_width

    fig.update_layout(**layout_args)

    return fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        config={'responsive': responsive, 'displayModeBar': display_mode_bar}
    )


def generic_plotly_pie_chart(
        labels: List[str],
        values: List[Union[int, float]],
        height: int = 400,
        width: int = 450,
        title: Optional[str] = None,
        colors: Optional[List[str]] = None,
        textinfo: str = 'percent+label',
        textfont_dict: Optional[Dict[str, Any]] = None,
        hoverinfo: str = 'label+percent+value',
        showlegend: bool = True,
        legend_dict: Optional[Dict[str, Any]] = None,
        margin_dict: Optional[Dict[str, int]] = None,
        paper_bgcolor: str = 'rgba(0,0,0,0)',
        plot_bgcolor: str = 'rgba(0,0,0,0)',
        font_dict: Optional[Dict[str, Any]] = None,
        sort_traces: bool = False,
        responsive: bool = True,
        display_mode_bar: bool = False,
        include_plotlyjs: bool = False,
        full_html: bool = False
) -> str:
    if textfont_dict is None:
        textfont_dict = dict(size=11)
    if legend_dict is None:
        legend_dict = dict(
            font=dict(size=9, family="Lato, Arial, Helvetica, sans-serif"),
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        )
    if margin_dict is None:
        margin_dict = dict(l=10, r=10, t=10, b=50 if showlegend else 20)
    if font_dict is None:
        font_dict = dict(family="Lato, Arial, Helvetica, sans-serif")

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo=textinfo,
        textfont=textfont_dict,
        hoverinfo=hoverinfo,
        sort=sort_traces,
        showlegend=showlegend
    )])

    fig.update_layout(
        title_text=title,
        height=height,
        width=width,
        margin=margin_dict,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        font=font_dict,
        legend=legend_dict
    )
    return fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        config={'responsive': responsive, 'displayModeBar': display_mode_bar}
    )


def generic_plotly_bar_chart(
        y_values: List[Union[str, int, float]],
        x_values: List[Union[int, float]],
        orientation: str,
        height: int = 400,
        width: int = 450,
        title: Optional[str] = None,
        bar_color: Union[str, List[str]] = '#003049',
        text_template: Optional[str] = None,
        textposition: str = 'outside',
        textfont_dict: Optional[Dict[str, Any]] = None,
        hoverinfo: str = 'x+y',  # Default, will be adapted by plotly if orientation changes
        margin_dict: Optional[Dict[str, int]] = None,
        paper_bgcolor: str = 'rgba(0,0,0,0)',
        plot_bgcolor: str = 'rgba(0,0,0,0)',
        xaxis_dict: Optional[Dict[str, Any]] = None,
        yaxis_dict: Optional[Dict[str, Any]] = None,
        bargap: float = 0.2,  # Slightly reduced default bargap for a tighter look
        font_dict: Optional[Dict[str, Any]] = None,
        responsive: bool = True,
        display_mode_bar: bool = False,
        include_plotlyjs: bool = False,
        full_html: bool = False
) -> str:
    if textfont_dict is None:
        textfont_dict = dict(size=9)
    if margin_dict is None:
        margin_dict = dict(l=80 if orientation == 'h' else 40, r=20, t=5, b=20)
    if font_dict is None:
        font_dict = dict(family="Lato, Arial, Helvetica, sans-serif")

    default_axis_config = dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(size=9, color="#333333")
    )
    if xaxis_dict is None:
        xaxis_dict = default_axis_config.copy()
    if yaxis_dict is None:
        yaxis_dict = default_axis_config.copy()
        if orientation == 'h':  # For horizontal bar charts, reverse y-axis category order
            yaxis_dict['autorange'] = "reversed"

    if orientation == 'h':
        data_params = dict(y=y_values, x=x_values)
        if text_template is None: text_template = "%{x:.2f}"
    elif orientation == 'v':
        data_params = dict(x=y_values, y=x_values)
        if text_template is None: text_template = "%{y:.2f}"
    else:
        raise ValueError("Orientation must be 'h' or 'v'")

    fig = go.Figure(data=[go.Bar(
        **data_params,
        orientation=orientation,
        marker_color=bar_color,
        texttemplate=text_template,
        textposition=textposition,
        textfont=textfont_dict,
        hoverinfo=hoverinfo
    )])

    fig.update_layout(
        title_text=title,
        height=height,
        width=width,
        margin=margin_dict,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        xaxis=xaxis_dict,
        yaxis=yaxis_dict,
        bargap=bargap,
        font=font_dict
    )
    return fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        config={'responsive': responsive, 'displayModeBar': display_mode_bar}
    )


def generic_plotly_grouped_bar_chart(
        x_values: List[str],
        series_data: List[Dict[str, Any]],
        chart_title: str,
        height: int,
        width: Optional[int] = None,
        y_axis_tick_format: Optional[str] = None,
        bar_text_template: Optional[str] = None,
        bar_text_position: str = "outside",
        bar_text_font_size_factor: float = 1.0,
        barmode: str = "group",
        legend_dict: Optional[Dict[str, Any]] = None,
        margin_dict: Optional[Dict[str, int]] = None,
        title_x_position: float = 0.05,
        xaxis_tickangle: Optional[float] = None,
        paper_bgcolor: str = 'rgba(0,0,0,0)',
        plot_bgcolor: str = 'rgba(0,0,0,0)',
        include_plotlyjs: bool = False,
        full_html: bool = False,
        display_mode_bar: bool = False,
        responsive: bool = True,
        theme_mode:ThemeMode =ThemeMode.light

) -> str:
    fig = go.Figure()


    styles=get_theme_settings(theme_mode)

    for series in series_data:
        trace = go.Bar(
            name=series['name'],
            x=x_values,
            y=series['y_values'],
            marker_color=series.get('color', None)
        )
        if bar_text_template:
            trace.texttemplate = bar_text_template
            trace.textposition = bar_text_position
            trace.textfont = dict(
                size=int(styles.chart_label_font_size * bar_text_font_size_factor),
                family=styles.font_family_paragraphs,
                color=styles.main_color
            )
        fig.add_trace(trace)

    default_legend_config = dict(
        font=dict(size=styles.chart_label_font_size, family=styles.font_family_paragraphs),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor='rgba(0,0,0,0)'
    )
    if legend_dict is not None:
        default_legend_config.update(legend_dict)

    final_margin_dict = margin_dict if margin_dict is not None else dict(l=40, r=20, t=50 if chart_title else 50, b=50)
    if xaxis_tickangle is not None and xaxis_tickangle != 0:
        final_margin_dict["b"] = max(final_margin_dict.get("b", 30), 70 + abs(xaxis_tickangle) // 10 * 5)


    fig.update_layout(
        title_text=chart_title,
        title_font=dict(size=styles.font_size_h4, family=styles.font_family_paragraphs, color=styles.main_color),
        title_x=title_x_position,
        height=height,
        width=width,
        barmode=barmode,
        xaxis_tickfont_size=styles.chart_label_font_size,
        yaxis_tickfont_size=styles.chart_label_font_size,
        yaxis_tickformat=y_axis_tick_format,
        legend=default_legend_config,
        margin=final_margin_dict,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        font=dict(family=styles.font_family_paragraphs)
    )



    if xaxis_tickangle is not None:
        fig.update_xaxes(tickangle=xaxis_tickangle)


    return fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        config={'responsive': responsive, 'displayModeBar': display_mode_bar}
    )


def title_card_template(
        slide_title: str,
        main_heading: str,
        sub_heading: Optional[str] = None,
        background_color: str = "#f8f9fa",
        title_font_size: int = 48,
        subtitle_font_size: int = 24,
        title_color: str = "#333333",
        subtitle_color: str = "#555555"
) -> Slide:
    elements = [
        TextElement(
            text=main_heading,
            font_size=title_font_size,
            font_weight=FontWeight.bold,
            color=title_color,
            h_align=HorizontalAlign.center,
            position=Position(top="35%", left="5%", right="5%")
        )
    ]
    if sub_heading:
        elements.append(
            TextElement(
                text=sub_heading,
                font_size=subtitle_font_size,
                color=subtitle_color,
                h_align=HorizontalAlign.center,
                position=Position(top="50%", left="10%", right="10%")
            )
        )
    layout = AbsoluteLayout(elements=elements, width="100%", height="100%")
    return Slide(title=slide_title, layout=layout, background_color=background_color)


def two_column_text_chart_template(
        slide_title_text: str,
        text_html_content: str,
        chart_function_name: str,
        chart_params: Dict[str, Any],
        text_col_definition: str = "1fr",
        chart_col_definition: str = "1.5fr",
        background_color: str = "#ffffff",
        gap: int = 20
) -> Slide:
    text_el = HtmlElement(
        html=f'<div style="padding: 10px; font-size: 18px; line-height: 1.7;">{text_html_content}</div>'
    )
    chart_el = FunctionElement(function=chart_function_name, params=chart_params)

    layout = GridLayout(
        row_definitions=["1fr"],
        col_definitions=[text_col_definition, chart_col_definition],
        gap=gap,
        cells=[
            GridCell(row=1, col=1, element=text_el, padding="10px", align_self="stretch"),
            GridCell(row=1, col=2, element=chart_el, padding="10px", align_self="stretch")
        ]
    )
    return Slide(title=slide_title_text, layout=layout, background_color=background_color)


def bullet_points_main_image_template(
        slide_title_text: str,
        image_src: str,
        image_alt: str = "Slide image",
        bullet_points_html: List[str] = [],
        background_color: str = "#ffffff",
        image_col_def: str = "1.8fr",
        text_col_def: str = "1fr",
        gap: int = 25
) -> Slide:
    image_el = ImageElement(
        src=image_src,
        alt=image_alt,
        size=Size(width="100%", height="100%"),
        object_fit="contain"
    )

    text_elements_html = "".join(
        [f'<div style="margin-bottom: 15px; font-size: 18px; line-height:1.6;">{point}</div>' for point in
         bullet_points_html])

    combined_text_el = HtmlElement(
        html=f'<div style="padding: 15px;">{text_elements_html}</div>'
    )

    layout = GridLayout(
        row_definitions=["1fr"],
        col_definitions=[image_col_def, text_col_def],
        gap=gap,
        cells=[
            GridCell(row=1, col=1, element=image_el, padding="10px", align_self="center", justify_self="center"),
            GridCell(row=1, col=2, element=combined_text_el, padding="10px", align_self="stretch")
        ]
    )
    return Slide(title=slide_title_text, layout=layout, background_color=background_color)