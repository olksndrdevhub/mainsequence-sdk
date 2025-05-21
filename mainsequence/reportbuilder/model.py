"""
Pydantic-based declarative slide framework — Python 3.9-friendly
----------------------------------------------------------------
* Pure Pydantic 2-style models (no nested Dict[…] fields)
* All CSS lengths are **strings** (e.g. "300px", "50%").  Integers are
  accepted at runtime and coerced to a px-string by validators – type
  hints stay simple (`Optional[str]`).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Type

from pydantic import BaseModel, Field, validator, ValidationError
from jinja2 import Environment

from typing import Callable, Dict

# Every function must accept a *Pydantic model instance* and return raw HTML
FUNCTION_REGISTRY: Dict[str, Callable] = {}

def register_function(name: str):
    """
    Decorator that registers `fn` under the given name.
    The decorated function **must** take one positional argument
    (a Pydantic model with the parameters) and return an HTML string.
    """
    def decorator(fn: Callable):
        if name in FUNCTION_REGISTRY:
            raise ValueError(f"Function '{name}' already registered")
        FUNCTION_REGISTRY[name] = fn
        return fn
    return decorator

# ────────────────────────────── Common enums ──────────────────────────────

class HorizontalAlign(str, Enum):
    left = "left"
    center = "center"
    right = "right"


class VerticalAlign(str, Enum):
    top = "top"
    middle = "middle"
    bottom = "bottom"


class FontWeight(str, Enum):
    normal = "normal"
    bold = "bold"


class Anchor(str, Enum):
    top_left = "top_left"
    top_right = "top_right"
    bottom_left = "bottom_left"
    bottom_right = "bottom_right"
    center = "center"


# ──────────────────────────────── Primitives ───────────────────────────────

class Size(BaseModel):
    width:  Optional[str] = None   # "300px", "40%", …
    height: Optional[str] = None

    @validator("width", "height", pre=True)
    def _coerce_to_str(cls, v):  # noqa: N805
        if v is None:
            return v
        return f"{v}px" if isinstance(v, int) else str(v)

    def css(self) -> str:
        return "".join(
            f"{dim}:{val};"
            for dim, val in (("width", self.width), ("height", self.height))
            if val is not None
        )


class Position(BaseModel):
    top:    Optional[str] = None
    left:   Optional[str] = None
    right:  Optional[str] = None
    bottom: Optional[str] = None
    anchor: Optional[Anchor] = None

    @validator("top", "left", "right", "bottom", pre=True)
    def _coerce_to_str(cls, v):  # noqa: N805
        if v is None:
            return v
        return f"{v}px" if isinstance(v, int) else str(v)

    def css(self) -> str:
        if self.anchor and any([self.top, self.left, self.right, self.bottom]):
            raise ValueError("Specify either ‘anchor’ or explicit offsets – not both")
        if self.anchor:
            return {
                Anchor.top_left:    "top:0;left:0;",
                Anchor.top_right:   "top:0;right:0;",
                Anchor.bottom_left: "bottom:0;left:0;",
                Anchor.bottom_right:"bottom:0;right:0;",
                Anchor.center:      "top:50%;left:50%;transform:translate(-50%,-50%);",
            }[self.anchor]

        return "".join(
            f"{side}:{val};"
            for side, val in (
                ("top", self.top), ("left", self.left),
                ("right", self.right), ("bottom", self.bottom)
            )
            if val is not None
        )


# ────────────────────────── Element hierarchy ────────────────────────────

class ElementBase(BaseModel):
    id:      str = Field(default_factory=lambda: f"elem_{id(object())}")
    z_index: int = 1

    class Config:
        arbitrary_types_allowed = True

    def render(self) -> str:   # noqa: D401, PLR0201
        raise NotImplementedError


class TextElement(ElementBase):
    text:         str
    font_size:    int = 16
    font_weight:  FontWeight = FontWeight.normal
    h_align:      HorizontalAlign = HorizontalAlign.left
    v_align:      VerticalAlign   = VerticalAlign.top
    color:        str = "#000"
    size:         Size = Field(default_factory=Size)
    position:     Optional[Position] = None

    def render(self) -> str:
        style = [
            "position:absolute;" if self.position else "",
            self.position.css() if self.position else "",
            self.size.css(),
            f"font-size:{self.font_size}px;",
            f"font-weight:{self.font_weight.value};",
            f"color:{self.color};",
            f"text-align:{self.h_align.value};",
        ]
        return f'<div id="{self.id}" style="{"".join(style)}">{self.text}</div>'


class ImageElement(ElementBase):
    src:        str
    alt:        str = ""
    size:       Size = Field(default_factory=lambda: Size(width="100%", height="auto"))
    position:   Optional[Position] = None
    object_fit: str = "contain"

    def render(self) -> str:
        style = [
            "position:absolute;" if self.position else "",
            self.position.css() if self.position else "",
            self.size.css(),
            f"object-fit:{self.object_fit};",
        ]
        return (
            f'<img id="{self.id}" src="{self.src}" alt="{self.alt}" '
            f'style="{"".join(style)}" crossOrigin="anonymous" />'
        )


class HtmlElement(ElementBase):
    html: str

    def render(self) -> str:   # noqa: D401
        return f'<div id="{self.id}">{self.html}</div>'


class FunctionElement(ElementBase):
    """
    A slide element whose visual output is produced by a registered function.
    """
    function: str                         # key in FUNCTION_REGISTRY
    params:   Dict[str, Any] = {}         # raw args coming from YAML

    def render(self) -> str:
        if self.function not in FUNCTION_REGISTRY:
            raise ValueError(f"Unknown function '{self.function}'")

        fn = FUNCTION_REGISTRY[self.function]
        args_model: Type[BaseModel] = fn.__annotations__.get('return_model')  # see below

        # Validate/parse params against the args model
        try:
            typed_args = args_model(**self.params)
        except ValidationError as e:
            raise ValueError(
                f"Invalid parameters for '{self.function}':\n{e}"
            ) from None

        # Generate HTML and return
        html: str = fn(typed_args)
        return html


# ─────────────────────────────── Layouts ──────────────────────────────────

class GridCell(BaseModel):
    row:       int
    col:       int
    row_span:  int = 1
    col_span:  int = 1
    element:   Union[TextElement, ImageElement, HtmlElement]

    @validator("row", "col", "row_span", "col_span")
    def _positive(cls, v):  # noqa: N805
        if v < 1:
            raise ValueError("row/col indexes are 1-based and ≥ 1")
        return v

class GridLayout(BaseModel):
    rows: int
    cols: int
    gap:  int = 10
    cells: List[GridCell]

    @validator("cells", each_item=True)
    def _within_grid(cls, cell, values):  # noqa: N805
        rows = values.get("rows")
        cols = values.get("cols")
        if rows and cell.row + cell.row_span - 1 > rows:
            raise ValueError("GridCell exceeds row count")
        if cols and cell.col + cell.col_span - 1 > cols:
            raise ValueError("GridCell exceeds column count")
        return cell

    def render(self) -> str:
        grid_style = (
            "display:grid;"
            f"grid-template-columns:repeat({self.cols},1fr);"
            f"grid-template-rows:repeat({self.rows},1fr);"
            f"gap:{self.gap}px;position:relative;width:100%;height:100%;"
        )
        html_parts: List[str] = [f'<div class="slide-grid" style="{grid_style}">']
        for cell in self.cells:
            cell_style = (
                f"grid-column:{cell.col}/span {cell.col_span};"
                f"grid-row:{cell.row}/span {cell.row_span};position:relative;"
            )
            html_parts.append(f'<div style="{cell_style}">{cell.element.render()}</div>')
        html_parts.append("</div>")
        return "".join(html_parts)


class AbsoluteLayout(BaseModel):
    elements: List[Union[TextElement, ImageElement, HtmlElement]]

    def render(self) -> str:
        return "".join(e.render() for e in self.elements)


# ────────────────────────── Slide & Presentation ──────────────────────────

class Slide(BaseModel):
    title:             str
    layout:            Union[GridLayout, AbsoluteLayout]
    background_color:  Optional[str] = "#ffffff"
    notes:             Optional[str] = None

    def render(self, slide_number: int, total: int, theme: "Theme") -> str:
        header_html = (
            f'<div class="slide-header"><div class="slide-title fw-bold">{self.title}</div>'
            f'{theme.logo_img_html()}</div>'
        )
        footer_html = (
            f'<div class="slide-footer"><div class="slide-date">{theme.current_date}</div>'
            f'<div class="slide-number">{slide_number} / {total}</div></div>'
        )
        return (
            f'<section class="slide" style="background-color:{self.background_color};">'
            f"{header_html}<div class='slide-body'>{self.layout.render()}</div>{footer_html}</section>"
        )


class Theme(BaseModel):
    logo_url:             Optional[str] = None
    font_family:          str = "Helvetica, Arial, sans-serif"
    base_font_size:       int = 14
    cover_background_url: Optional[str] = None
    title_color:          str = "#000"
    current_date:         str = Field(default_factory=lambda: "{{CURRENT_DATE}}")

    def cover_html(self, presentation_title: str, subtitle: Optional[str]) -> str:
        bg     = f'<img class="cover-bg" src="{self.cover_background_url}" alt="cover" crossOrigin="anonymous">' \
                 if self.cover_background_url else ""
        sub    = f'<div class="cover-subtitle">{subtitle}</div>' if subtitle else ""
        logo   = self.logo_img_html(position="cover-logo")
        return f'<section class="slide cover-slide">{bg}<div><div class="cover-title">{presentation_title}</div>{sub}</div>{logo}</section>'

    def logo_img_html(self, position: str = "slide-logo") -> str:
        return f'<div class="{position}"><img src="{self.logo_url}" alt="logo" crossOrigin="anonymous"></div>' if self.logo_url else ""


class Presentation(BaseModel):
    title:    str
    subtitle: Optional[str] = None
    slides:   List[Slide]
    theme:    Theme = Field(default_factory=Theme)

    def render(self) -> str:
        slides_html = [self.theme.cover_html(self.title, self.subtitle)]
        total = len(self.slides)
        slides_html += [s.render(i + 1, total, self.theme) for i, s in enumerate(self.slides)]
        return BASE_TEMPLATE.render(
            title         = self.title,
            font_family   = self.theme.font_family,
            base_font_size= self.theme.base_font_size,
            title_color   = self.theme.title_color,
            slides        = "".join(slides_html),
        )


# ───────────────────────────── HTML template ─────────────────────────────

base_template_str = (Path("templates") / "base_template.html").read_text()
BASE_TEMPLATE = Environment().from_string(base_template_str)
