from model import (
    Presentation,
    Theme,
    Slide,
    GridLayout,
    GridCell,
    TextElement,
    FunctionElement,
    HorizontalAlign,
    VerticalAlign,
    FontWeight
)
import yaml
from pathlib import Path
from model import Presentation

def create_presentation_from_code() -> None:

    # Define the Theme
    theme_obj = Theme(
        logo_url="https://cdn.prod.website-files.com/67d166ea95c73519badbdabd/67d166ea95c73519badbdc60_Asset%25202%25404x-8-p-800.png",
        current_date="May 21, 2025",
        font_family="Lato, Arial, Helvetica, sans-serif",
        base_font_size=20,
        title_color="#005f73"
    )

    # Define Slide 1: Welcome: Strategic Insights & Focus
    slide1_cells = [
        GridCell(
            row=1, col=1,
            element=TextElement(
                text="Visualizing Impact",
                font_size=46,
                font_weight=FontWeight.bold,
                color="#003049",
                h_align=HorizontalAlign.left,
                v_align=VerticalAlign.bottom
            )
        ),
        GridCell(
            row=2, col=1,
            element=TextElement(
                text="Data-driven insights for a clearer tomorrow.",
                font_size=22,
                font_weight=FontWeight.normal,
                color="#005f73",
                h_align=HorizontalAlign.left,
                v_align=VerticalAlign.top
            )
        ),
        GridCell(
            row=3, col=1, row_span=2,
            element=TextElement(
                text="""<p style="text-align: left; font-size: 20px; color: #333333; margin-bottom: 10px;">Our core strategic segments focus on holistic growth and resilience:</p>
<ul style="text-align: left; margin: 0; padding-left: 25px; font-size: 19px; line-height: 1.7;">
  <li><b>Strategic Alignment:</b> Ensuring initiatives drive core objectives and long-term vision.</li>
  <li><b>Operational Excellence:</b> Optimizing processes for maximum efficiency and quality.</li>
  <li><b>Market Innovation:</b> Pioneering solutions and exploring emerging opportunities.</li>
  <li><b>Future Readiness:</b> Building resilient frameworks to adapt to evolving landscapes.</li>
</ul>
<p style="text-align: left; font-size: 20px; color: #555555; margin-top: 15px;">These balanced segments ensure we are making data understandable and actionable across the organization.</p>""",
                font_weight=FontWeight.normal,
                h_align=HorizontalAlign.left,
                v_align=VerticalAlign.middle,  # Assuming model.py GridLayout.render uses this
                color="#333333"
            )
        ),
        GridCell(
            row=1, col=2, row_span=4,
            element=FunctionElement(
                function="pie_chart",
                params={
                    "title": "Key Strategic Segments",
                    "labels": ["Strategic Alignment", "Operational Excellence", "Market Innovation", "Future Readiness"],
                    "values": [10, 10, 10, 70],
                    "height": 500
                }
            )
        )
    ]

    slide1 = Slide(
        title="Welcome: Strategic Insights & Focus",
        background_color="#ffffff",
        layout=GridLayout(
            rows=4,
            cols=2,
            gap=25,
            cells=slide1_cells
        )
    )

    # Define Slide 2: Project Horizon: Performance Review
    slide2_cells = [
        GridCell(
            row=1, col=1,
            element=FunctionElement(
                function="line_chart",
                params={
                    "title": "Quarterly Performance Index",
                    "x": ["Q1 '24", "Q2 '24", "Q3 '24", "Q4 '24", "Q1 '25"],
                    "y": [2.8, 3.2, 3.1, 3.5, 4.0],
                    "series_name": "Key Performance Index",
                    "height": 320,
                    "labels": {"x": "Quarter", "y": "Index Value"},
                    "markers": True
                }
            )
        ),
        GridCell(
            row=1, col=2,
            element=TextElement(
                text="""<ul style="list-style-position: outside; margin: 0; padding-left: 20px; text-align: left;">
  <li><b>Consistent Upward Trend:</b> The Key Performance Index for Project Horizon has shown steady growth.</li>
  <li>This indicates increasing positive impact and successful milestone achievement quarter over quarter.</li>
</ul>""",
                font_size=20,
                font_weight=FontWeight.normal,
                line_height=1.7,  # Note: line_height is not in the standard TextElement model, add if needed
                color="#333333",
                h_align=HorizontalAlign.left,
                v_align=VerticalAlign.middle  # Assuming model.py GridLayout.render uses this
            )
        ),
        GridCell(
            row=2, col=1,
            element=FunctionElement(
                function="bar_chart",
                params={
                    "title": "Strategic Goal Achievement (%)",
                    "x": ["Innov.", "Effic.", "Reach"],
                    "y": [85, 70, 92],
                    "orientation": "v",
                    "height": 320,
                    "labels": {"x": "Pillar", "y": "Achievement (%)"}
                }
            )
        ),
        GridCell(
            row=2, col=2,
            element=TextElement(
                text="""<ul style="list-style-position: outside; margin: 0; padding-left: 20px; text-align: left;">
  <li><b>Meeting Key Targets:</b> Project Horizon excels in its strategic pillars.</li>
  <li>Innovation is at 85%, Efficiency at 70%, and Market Reach has expanded to 92%, reflecting strong execution.</li>
</ul>""",
                font_size=20,
                font_weight=FontWeight.normal,
                line_height=1.7,  # Note: line_height is not in the standard TextElement model, add if needed
                color="#333333",
                h_align=HorizontalAlign.left,
                v_align=VerticalAlign.middle  # Assuming model.py GridLayout.render uses this
            )
        )
    ]

    slide2 = Slide(
        title="Project Horizon: Performance Review",
        background_color="#ffffff",
        layout=GridLayout(
            rows=2,
            cols=2,
            gap=35,
            cells=slide2_cells
        )
    )

    # Create the Presentation object
    presentation_obj = Presentation(
        title="Visualizing Tomorrow's Insights",
        subtitle="A Showcase of Dynamic Data Presentation",
        theme=theme_obj,
        slides=[slide1, slide2]
    )

    return presentation_obj



def load_and_render(slides_name: str) -> None:
    """
    Load a presentation definition from a YAML file and render it to HTML.
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    yaml_path = Path("configurations") / f"{slides_name}.yaml"
    output_html_path = output_dir / f"{slides_name}.html"
    output_pdf_path = output_dir / f"{slides_name}.pdf"

    raw = Path(yaml_path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    pres = Presentation(**data)
    html = pres.render()
    Path(output_html_path).write_text(html, encoding="utf-8")
    print(f"Rendered presentation to {output_html_path}")
    # from weasyprint import HTML, CSS
    #
    # html_doc = HTML(string=html, base_url=str(output_dir.resolve()))
    # html_doc.write_pdf(output_pdf_path)

if __name__ == "__main__":
    load_and_render(slides_name="actinver_example")


    # my_presentation = create_presentation_from_code()
    # try:
    #     with open("presentation_from_code.html", "w", encoding="utf-8") as f:
    #         f.write(my_presentation.render())
    #     print("Presentation rendered to presentation_from_code.html")
    # except Exception as e:
    #     print(f"An error occurred during rendering: {e}")