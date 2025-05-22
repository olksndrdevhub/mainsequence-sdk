from typing import List, Dict, Any, Optional, Callable
from mainsequence.reportbuilder.model import (
    Slide,
    GridLayout, GridCell, AbsoluteLayout,
    TextElement, FunctionElement, HtmlElement, ImageElement,
    HorizontalAlign, VerticalAlign, FontWeight,
    Size, Position, Anchor
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
            position=Position(top="35%", left="5%", right="5%", width="90%")
        )
    ]
    if sub_heading:
        elements.append(
            TextElement(
                text=sub_heading,
                font_size=subtitle_font_size,
                color=subtitle_color,
                h_align=HorizontalAlign.center,
                position=Position(top="50%", left="10%", right="10%", width="80%")
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
    text_el = HtmlElement(type="HtmlElement",
                          html=f'<div style="padding: 10px; font-size: 18px; line-height: 1.7;">{text_html_content}</div>')
    chart_el = FunctionElement(type="FunctionElement", function=chart_function_name, params=chart_params)

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
        bullet_points_html: List[str] = [],  # List of HTML strings for each bullet point section
        background_color: str = "#ffffff",
        image_col_def: str = "1.8fr",
        text_col_def: str = "1fr",
        gap: int = 25
) -> Slide:
    image_el = ImageElement(
        src=image_src,
        alt=image_alt,
        size=Size(width="100%", height="100%"),  # Cell will control effective size via flex
        object_fit="contain"  # or "cover"
    )

    text_elements_html = "".join(
        [f'<div style="margin-bottom: 15px; font-size: 18px; line-height:1.6;">{point}</div>' for point in
         bullet_points_html])

    combined_text_el = HtmlElement(
        html=f'<div style="padding: 15px;">{text_elements_html}</div>'
    )

    layout = GridLayout(
        row_definitions=["1fr"],
        col_definitions=[image_col_def, text_col_def],  # Image left, text right
        gap=gap,
        cells=[
            GridCell(row=1, col=1, element=image_el, padding="10px", align_self="center", justify_self="center"),
            GridCell(row=1, col=2, element=combined_text_el, padding="10px", align_self="stretch")
        ]
    )
    return Slide(title=slide_title_text, layout=layout, background_color=background_color)




## PLOTLY ##
def _transpose_for_plotly(data_rows: List[List[Any]], num_columns: int) -> List[List[Any]]:
    if not data_rows:
        return [[] for _ in range(num_columns)]
    transposed = list(map(list, zip(*data_rows)))
    return transposed

def create_plotly_table_html(headers, rows, table_height, column_widths=None, cell_align=None, fig_width=None,
                             responsive_config=True, header_visible=True):
    header_font_size = 10
    cell_font_size = 9
    header_fill_color = 'rgb(0, 32, 96)'
    header_font_color = 'white'
    cell_fill_color = 'rgb(240,245,255)'
    line_color = 'rgb(200,200,200)'

    shared_header_align = "center"

    plotly_column_data = _transpose_for_plotly(rows, len(headers))

    header_config = dict(
        values=headers, fill_color=header_fill_color,
        font=dict(color=header_font_color, size=header_font_size if header_visible else 0),
        align=shared_header_align, line_color=line_color,
        height=22 if header_visible else 0
    )

    fig = go.Figure(data=[go.Table(
        header=header_config,
        cells=dict(
            values=plotly_column_data, fill_color=cell_fill_color,
            font=dict(size=cell_font_size),  # Single font dict for all cells
            align=cell_align or 'left', line_color=line_color, height=20
        ),
        columnwidth=column_widths if column_widths else []
    )])

    layout_args = {
        "height": table_height,
        "margin": dict(l=5, r=5, t=2, b=2),
        "paper_bgcolor": 'rgba(0,0,0,0)',
        "plot_bgcolor": 'rgba(0,0,0,0)'
    }
    if fig_width:
        layout_args["width"] = fig_width

    fig.update_layout(**layout_args)

    return fig.to_html(include_plotlyjs=False, full_html=False,
                       config={'responsive': responsive_config, 'displayModeBar': False})
