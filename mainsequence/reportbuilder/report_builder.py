from model import (
    Anchor, GridCell, GridLayout, ImageElement,
    Presentation, Size, Slide, TextElement, Theme, AbsoluteLayout
)


def main() -> None:
    # Slide 1 ──────────────────────────────────────────
    slide1 = Slide(
        title="Welcome",
        layout=AbsoluteLayout(
            elements=[
                TextElement(
                    text="Big Welcome!",
                    font_size=30,
                    font_weight="bold",
                    position={"anchor": Anchor.center},
                )
            ]
        ),
    )

    # Slide 2 ──────────────────────────────────────────
    cells = []
    for idx, label in enumerate(("Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"), start=1):
        row = 1 if idx <= 2 else 2
        col = 1 if idx % 2 else 2
        cells.append(
            GridCell(
                row=row, col=col,
                element=TextElement(text=label, font_size=18, h_align="center", v_align="middle"),
            )
        )
    slide2 = Slide(title="Grid Example", layout=GridLayout(rows=2, cols=2, cells=cells))

    # Slide 3 ──────────────────────────────────────────
    slide3 = Slide(
        title="Image + Text",
        layout=GridLayout(
            rows=1, cols=2,
            cells=[
                GridCell(
                    row=1, col=1,
                    element=ImageElement(
                        src="https://placehold.co/400x300",
                        size=Size(width="100%", height="100%"),
                        object_fit="cover",
                    ),
                ),
                GridCell(
                    row=1, col=2,
                    element=TextElement(
                        text="Account overview chart on the right.",
                        font_size=20,
                        h_align="center",
                    ),
                ),
            ],
        ),
    )

    pres = Presentation(
        title="Demo Presentation",
        subtitle="Automated with Pydantic",
        slides=[slide1, slide2, slide3],
        theme=Theme(
            logo_url="https://cdn.prod.website-files.com/67d166ea95c7351…ea95c73519badbdc60_Asset%25202%25404x-8-p-800.png",
            current_date="21-May-25",
        ),
    )
    with open("demo_output.html", "w", encoding="utf-8") as fh:
        fh.write(pres.render())
    print("demo_output.html written")


# render_presentation.py

import yaml
from pathlib import Path
from model import Presentation

def load_and_render(slides_name: str) -> None:
    """
    Load a presentation definition from a YAML file and render it to HTML.
    """
    yaml_path = Path("configurations") / f"{slides_name}.yaml"
    output_path = Path("output") / f"{slides_name}.html"

    raw = Path(yaml_path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    pres = Presentation(**data)
    html = pres.render()
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Rendered presentation to {output_path}")

if __name__ == "__main__":
    load_and_render(slides_name="example_simple")