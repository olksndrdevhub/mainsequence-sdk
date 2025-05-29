
import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from operator import attrgetter
import pandas as pd
from babel.dates import format_date
import numpy as np
import json
from mainsequence.reportbuilder.model import (    Presentation,    Slide, VerticalImageSlide,    GridLayout,
    GridCell,    TextElement,TextH1,TextH2,    ImageElement,    HtmlElement,    FontWeight,    HorizontalAlign,
    VerticalAlign,    Size, update_settings_from_dict, ThemeMode, light_settings, dark_settings,)
from mainsequence.reportbuilder.slide_templates import (
    generic_plotly_table,    generic_plotly_pie_chart,    generic_plotly_bar_chart, generic_plotly_grouped_bar_chart)
from mainsequence.client import (Account, Asset, AccountHistoricalHoldings)
from mainsequence.client.models_helpers import MarketsTimeSeriesDetails
from mainsequence.tdag import TimeSerie, APITimeSerie

update_settings_from_dict(
    overrides=dict(
        primary_color="#041e42",
        heading_color="#303238",
        font_size_h1=52,
        chart_palette_categorical= ["#041e42", "#aea06c", "#808080", "#e7298a", "#66a61e"],
        logo_url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/2560px-Google_2015_logo.svg.png",
    ), mode=ThemeMode.light,
)

update_settings_from_dict(
    overrides=dict(
        background_color= "#041e42",
        logo_url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/2560px-Google_2015_logo.svg.png",
    ), mode=ThemeMode.dark,
)



def title_slide():
    el_main_title = TextH1(   font_weight=FontWeight.bold, text="Titke",    )
    el_subtitle = TextH2(  text= f"subtitle",)
    el_logo = ImageElement(
        src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/2560px-Google_2015_logo.svg.png",
        alt=" Logo",
        size=Size(height="30px", width="auto")
    )

    el_tagline = TextElement(   text=f"Tag Line Example" )

    cover_layout = GridLayout(
        row_definitions=["20%", "auto", "auto", "auto", "auto", ],
        col_definitions=["100%"],
        cells=[
            GridCell(row=2, col=1, element=el_main_title, padding="0 0 0 50px"  ),
            GridCell(row=3, col=1, element=el_subtitle,  padding="0 0 0 50px", ),
            GridCell(row=4, col=1, element=el_tagline,   padding="0 0 0 50px" ),
            GridCell(row=5, col=1, element=el_logo,     padding="125px 0 0 50px"),
        ],
        width="100%",
        height="100%"
    )

    return VerticalImageSlide(
        title="",
        layout=cover_layout,
        include_logo_in_header=False,
        style_theme=dark_settings,
        image_width_pct=40,
        image_url="https://cdn.prod.website-files.com/67d166ea95c73519badbdabd/67d17edc6ef90726f9c4af9b_office.png"
    )


def build_test_slide():
    cells = []
    row_definitions,col_definitions=[],[]
    for i  in range(20):
        cells += [
            GridCell(
                row=i+1, col=1,
                element=TextElement(element_type="h4", text=f" Text {i}", v_align=VerticalAlign.center),
                padding="2px 10px 2px 0", align_self="start"
            ),

        ]
        row_definitions.append("auto")
        col_definitions.append("auto")




    slide_layout = GridLayout(
        row_definitions=row_definitions,
        col_definitions=col_definitions,
        cells=cells
    )

    return Slide(

        title=f"Slide Title ",
        layout=slide_layout,
        footer_info=f"Footer"
    )


if __name__ == "__main__":

    presentation = Presentation(
        title="Example Presentation",
        slides=[
            title_slide(),
            build_test_slide(),
            build_test_slide()

        ]
    )

    try:
        html_content = presentation.render()
        script_dir = Path(__file__).resolve().parent
        output_html_path = script_dir / "template_report.html"
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Presentation rendered successfully to local storage {output_html_path.resolve()}")

        # Handle and display any errors during rendering
    except Exception as e:
        print(f"An error occurred during rendering: {e}")
        import traceback

        traceback.print_exc()


