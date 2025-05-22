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

def load_and_render(slides_name: str) -> None:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = Path("configurations") / f"{slides_name}.yaml"
    output_html_path = output_dir / f"{slides_name}.html"
    output_pdf_path = output_dir / f"{slides_name}.pdf"

    raw = Path(yaml_path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    pres = Presentation(**data)
    html = pres.render()
    Path(output_html_path).write_text(html, encoding="utf-8")
    print(f"Rendered presentation to {output_html_path}")

if __name__ == "__main__":
    try:
        with open("presentation_from_code.html", "w", encoding="utf-8") as f:
            f.write(my_presentation.render())
        print("Presentation rendered to presentation_from_code.html")
    except Exception as e:
        print(f"An error occurred during rendering: {e}")