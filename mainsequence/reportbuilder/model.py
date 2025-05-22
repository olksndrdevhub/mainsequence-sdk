from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Type, Any

from pydantic import BaseModel, Field, validator, ValidationError
from jinja2 import Environment

from typing import Callable, Dict

from mainsequence.reportbuilder.functions import FUNCTION_REGISTRY


# ────────────────────────────── Common enums ──────────────────────────────

class HorizontalAlign(str, Enum):
    left = "left"
    center = "center"
    right = "right"

class VerticalAlign(str, Enum):
    top = "top"
    center = "center"
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
    def _coerce_to_str(cls, v):
        if v is None:
            return v
        return f"{v}px" if isinstance(v, int) else str(v)

    def css(self) -> str:
        if self.anchor and any([self.top, self.left, self.right, self.bottom]):
            raise ValueError("Specify either 'anchor' or explicit offsets - not both")
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

class ElementBase(BaseModel):
    id: str = Field(default_factory=lambda: f"elem_{id(object())}")
    z_index: int = 1
    css_class: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def render(self) -> str:
        raise NotImplementedError

class TextElement(ElementBase):
    text: str
    font_size: int = 16
    font_weight: FontWeight = FontWeight.normal
    h_align: HorizontalAlign = HorizontalAlign.left
    v_align: VerticalAlign = VerticalAlign.top
    color: str = "#000"
    line_height: Optional[str] = None
    size: Size = Field(default_factory=Size)
    position: Optional[Position] = None

    def render(self) -> str:
        style = []
        if self.position:
            style.append("position:absolute;")
            style.append(self.position.css())
        style.append(self.size.css())
        style.append(f"font-size:{self.font_size}px;")
        style.append(f"font-weight:{self.font_weight.value};")
        style.append(f"color:{self.color};")
        style.append(f"text-align:{self.h_align.value};")
        if self.line_height:
            style.append(f"line-height:{self.line_height};")
        
        class_attr = f'class="{self.css_class}"' if self.css_class else ""
        return f'<div id="{self.id}" {class_attr} style="{"".join(style)}">{self.text}</div>'

class ImageElement(ElementBase):
    src: str
    alt: str = ""
    size: Size = Field(default_factory=lambda: Size(width="100%", height="auto"))
    position: Optional[Position] = None
    object_fit: str = "contain"

    def render(self) -> str:
        style = []
        if self.position:
            style.append("position:absolute;")
            style.append(self.position.css())
        style.append(self.size.css())
        style.append(f"object-fit:{self.object_fit};")
        
        class_attr = f'class="{self.css_class}"' if self.css_class else ""
        return (
            f'<img id="{self.id}" {class_attr} src="{self.src}" alt="{self.alt}" '
            f'style="{"".join(style)}" crossOrigin="anonymous" />'
        )

class HtmlElement(ElementBase):
    html: str

    def render(self) -> str:
        class_attr = f'class="{self.css_class}"' if self.css_class else ""
        return f'<div id="{self.id}" {class_attr}">{self.html}</div>'

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
        args_model_field = fn.__annotations__.get('args') 
        if args_model_field is None :
            args_model_field = fn.__annotations__.get('return_model')

        args_model: Type[BaseModel] = args_model_field

        try:
            typed_args = args_model(**self.params) if args_model else self.params
        except ValidationError as e:
            raise ValueError(
                f"Invalid parameters for '{self.function}':\n{e}"
            ) from None
        
        html_output: str = fn(typed_args) if args_model else fn(**self.params)

        if self.css_class:
            return f'<div class="{self.css_class}">{html_output}</div>'
        return html_output

BaseElements = Union[TextElement, ImageElement, HtmlElement, FunctionElement]

class GridCell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    element: BaseElements
    padding: Optional[str] = None
    background_color: Optional[str] = None
    align_self: Optional[str] = None
    justify_self: Optional[str] = None

    @validator("row", "col", "row_span", "col_span", pre=True)
    def _positive(cls, v):
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        if not isinstance(v, int) or v < 1:
            raise ValueError("row/col/row_span/col_span must be positive integers >= 1")
        return v


class GridLayout(BaseModel):
    row_definitions: List[str] = Field(default_factory=lambda: ["1fr"])
    col_definitions: List[str] = Field(default_factory=lambda: ["1fr"])
    gap: int = 10
    cells: List[GridCell]
    width: Optional[str] = "100%"
    height: Optional[str] = "100%"

    @validator("gap",pre=True)
    def _coerce_gap_to_int(cls,v):
        if isinstance(v,str) and v.endswith("px"):
            return int(v[:-2])
        if isinstance(v,str) and v.isdigit():
            return int(v)
        if isinstance(v,int):
            return v
        raise ValueError("gap must be an int or string like '10px'")


    @validator("cells", each_item=True)
    def _within_grid(cls, cell: GridCell, values: Dict[str, Any]) -> GridCell:
        row_defs = values.get("row_definitions")
        col_defs = values.get("col_definitions")
        if row_defs and cell.row + cell.row_span - 1 > len(row_defs):
            raise ValueError(f"GridCell definition (row={cell.row}, row_span={cell.row_span}) exceeds row count ({len(row_defs)})")
        if col_defs and cell.col + cell.col_span - 1 > len(col_defs):
            raise ValueError(f"GridCell definition (col={cell.col}, col_span={cell.col_span}) exceeds column count ({len(col_defs)})")
        return cell
    
    def render(self) -> str:
        grid_style_parts = [
            "display:grid;",
            f"grid-template-columns:{' '.join(self.col_definitions)};",
            f"grid-template-rows:{' '.join(self.row_definitions)};",
            f"gap:{self.gap}px;",
            "position:relative;"
        ]
        if self.width:
            grid_style_parts.append(f"width:{self.width};")
        if self.height:
            grid_style_parts.append(f"height:{self.height};")
        grid_style = "".join(grid_style_parts)

        html_parts: List[str] = [f'<div class="slide-grid" style="{grid_style}">']
        for cell in self.cells:
            cell_styles_list = [
                f"grid-column:{cell.col}/span {cell.col_span};",
                f"grid-row:{cell.row}/span {cell.row_span};",
                "position:relative;",
                "display:flex;",
            ]

            align_items_css_value = "flex-start" 
            justify_content_css_value = "flex-start"

            if isinstance(cell.element, TextElement):
                if cell.element.v_align == VerticalAlign.center:
                    align_items_css_value = "center"
                elif cell.element.v_align == VerticalAlign.bottom:
                    align_items_css_value = "flex-end"
                
                if cell.element.h_align == HorizontalAlign.center:
                    justify_content_css_value = "center"
                elif cell.element.h_align == HorizontalAlign.right:
                    justify_content_css_value = "flex-end"
            
            cell_styles_list.append(f"align-items: {align_items_css_value};")
            cell_styles_list.append(f"justify-content: {justify_content_css_value};")

            if cell.padding:
                cell_styles_list.append(f"padding:{cell.padding};")
            if cell.background_color:
                cell_styles_list.append(f"background-color:{cell.background_color};")
            if cell.align_self:
                cell_styles_list.append(f"align-self:{cell.align_self};")
            if cell.justify_self:
                cell_styles_list.append(f"justify-self:{cell.justify_self};")

            final_cell_style = "".join(cell_styles_list)
            html_parts.append(f'<div style="{final_cell_style}">{cell.element.render()}</div>')
        html_parts.append("</div>")
        return "".join(html_parts)

class AbsoluteLayout(BaseModel):
    elements: List[BaseElements]
    width: Optional[str] = "100%"
    height: Optional[str] = "100%"

    def render(self) -> str:
        style_parts = ["position:relative;"]
        if self.width:
            style_parts.append(f"width:{self.width};")
        if self.height:
            style_parts.append(f"height:{self.height};")
        
        style = "".join(style_parts)
        elements_html = "".join(e.render() for e in self.elements)
        return f'<div style="{style}">{elements_html}</div>'

class Slide(BaseModel):
    title: str
    layout: Union[GridLayout, AbsoluteLayout]
    background_color: Optional[str] = "#ffffff"
    notes: Optional[str] = None

    def render(self, slide_number: int, total: int, theme: "Theme") -> str:
        header_html = (
            f'<div class="slide-header"><div class="slide-title fw-bold">{self.title}</div>'
            f'{theme.logo_img_html()}</div>'
        )
        footer_html = (
            f'<div class="slide-footer"><div class="slide-date">{theme.current_date}</div>'
            f'<div class="slide-number">{slide_number} / {total}</div></div>'
        )
        slide_body_style = "flex:1;display:flex;flex-direction:column;overflow:hidden;"

        return (
            f'<section class="slide" style="background-color:{self.background_color};">'
            f"{header_html}<div class='slide-body' style='{slide_body_style}'>{self.layout.render()}</div>{footer_html}</section>"
        )

class Theme(BaseModel):
    logo_url: Optional[str] = None
    font_family: str = "Helvetica, Arial, sans-serif"
    base_font_size: int = 14
    cover_background_url: Optional[str] = None
    title_color: str = "#000"
    current_date: str = Field(default_factory=lambda: "{{CURRENT_DATE}}")

    def cover_html(self, presentation_title: str, subtitle: Optional[str]) -> str:
        bg = f'<img class="cover-bg" src="{self.cover_background_url}" alt="cover" crossOrigin="anonymous">' \
             if self.cover_background_url else ""
        sub = f'<div class="cover-subtitle">{subtitle}</div>' if subtitle else ""
        logo = self.logo_img_html(position="cover-logo")
        return f'<section class="slide cover-slide">{bg}<div><div class="cover-title">{presentation_title}</div>{sub}</div>{logo}</section>'

    def logo_img_html(self, position: str = "slide-logo") -> str:
        return f'<div class="{position}"><img src="{self.logo_url}" alt="logo" crossOrigin="anonymous"></div>' if self.logo_url else ""

class Presentation(BaseModel):
    title: str
    subtitle: Optional[str] = None
    slides: List[Slide]
    theme: Theme = Field(default_factory=Theme)

    def render(self) -> str:
        slides_html = [self.theme.cover_html(self.title, self.subtitle)]
        total = len(self.slides)
        slides_html += [s.render(i + 1, total, self.theme) for i, s in enumerate(self.slides)]
        return BASE_TEMPLATE.render(
            title=self.title,
            font_family=self.theme.font_family,
            base_font_size=self.theme.base_font_size,
            title_color=self.theme.title_color,
            slides="".join(slides_html),
        )

base_template_str = (
    Path(__file__).parent
    / "base_templates"
    / "base_template.html"
).read_text(encoding="utf-8")
BASE_TEMPLATE = Environment().from_string(base_template_str)