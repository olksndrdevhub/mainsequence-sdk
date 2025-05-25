from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Type, Any,Literal

from pydantic import BaseModel, Field, validator, ValidationError,root_validator
from jinja2 import Environment

from typing import Callable, Dict

from mainsequence.reportbuilder.functions import FUNCTION_REGISTRY
from pydantic import  HttpUrl


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
    content_v_align: Optional[VerticalAlign] = None # For vertical alignment of content WITHIN the cell
    content_h_align: Optional[HorizontalAlign] = None # For horizontal alignment of content WITHIN the cell

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
            if cell.content_v_align:
                if cell.content_v_align == VerticalAlign.center:
                    align_items_css_value = "center"
                elif cell.content_v_align == VerticalAlign.bottom:
                    align_items_css_value = "flex-end"
                elif cell.content_v_align == VerticalAlign.top:
                    align_items_css_value = "flex-start"
            elif isinstance(cell.element, TextElement) and cell.element.v_align:
                if cell.element.v_align == VerticalAlign.center:
                    align_items_css_value = "center"
                elif cell.element.v_align == VerticalAlign.bottom:
                    align_items_css_value = "flex-end"

            justify_content_css_value = "flex-start"
            if cell.content_h_align:
                if cell.content_h_align == HorizontalAlign.center:
                    justify_content_css_value = "center"
                elif cell.content_h_align == HorizontalAlign.right:
                    justify_content_css_value = "flex-end"
                elif cell.content_h_align == HorizontalAlign.left:
                    justify_content_css_value = "flex-start"
            elif isinstance(cell.element, TextElement) and cell.element.h_align:
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
    layout: Union["GridLayout", "AbsoluteLayout"]
    background_color: str = Field(default_factory=lambda: light_settings.background_color)
    notes: Optional[str] = None
    title_font_size: int = 24
    body_margin_top: int = 40
    include_logo_in_header: bool = True
    footer_text_color: str =  Field(default_factory=lambda: light_settings.main_color)

    def _section_style(self) -> str:
        # only background color; size determined by container
        return f"background-color:{self.background_color};"

    def _render_header(self, theme: "Theme") -> str:
        title_style = f"font-size: {self.title_font_size}px;"
        logo_html = theme.logo_img_html() if self.include_logo_in_header else ""
        return (
            f'<div class="slide-header">'
            f'<div class="slide-title fw-bold" style="{title_style}">{self.title}</div>'
            f'{logo_html}'
            f'</div>'
        )

    def _render_body(self) -> str:
        style = (
            f"flex:1; display:flex; flex-direction:column; overflow:hidden;"
            f" margin-top:{self.body_margin_top}px;"
        )
        return (
            f'<div class="slide-body" style="{style}">'
            f'{self.layout.render()}'
            f'</div>'
        )

    def _render_footer(self, slide_number: int, total: int, theme: "Theme") -> str:
        text_style = f"color: {self.footer_text_color};"
        return (
            f'<div class="slide-footer">'
            f'<div class="slide-date" style="{text_style}">{theme.current_date}</div>'
            f'<div class="slide-number" style="{text_style}">{slide_number} / {total}</div>'
            f'</div>'
        )

    def render(self, slide_number: int, total: int, theme: "Theme") -> str:
        header = self._render_header(theme)
        body = self._render_body()
        footer = self._render_footer(slide_number, total, theme)
        section_style = self._section_style()

        return (
            f'<section class="slide" style="{section_style}">'
            f'{header}{body}{footer}'
            f'</section>'
        )


class VerticalImageSlide(Slide):
    image_url: HttpUrl = Field(
        ..., description="URL for the right-column image"
    )
    image_width_pct: int = Field(
        50,
        ge=0,
        le=100,
        description="Percentage width of the right-column image"
    )
    image_fit: Literal["cover", "contain"] = Field(
        "cover",
        description="How the image should fit its container"
    )

    def render(self, slide_number: int, total: int, theme: "Theme") -> str:
        header = self._render_header(theme)
        body = self._render_body()
        footer = self._render_footer(slide_number, total, theme)

        # Determine inline widths
        left_pct = 100 - self.image_width_pct
        left_style = f"width:{left_pct}%;"
        right_style = f"width:{self.image_width_pct}%; padding:0;"
        img_style = f"width:100%; height:100%; object-fit:{self.image_fit};"

        # Compose columns
        left_html = (
            f'<div class="left-column" style="{left_style}">'
            f'{body}'
            f'</div>'
        )
        right_html = (
            f'<div class="right-column" style="{right_style}">'
            f'  <img src="{self.image_url}" alt="" style="{img_style}" />'
            f'</div>'
        )

        # Section tag uses both classes and background style
        section_style = self._section_style()
        return (
            f'<section class="slide vertical-image-slide" style="{section_style}">'
            f'{left_html}{right_html}'
            f'</section>'
        )


class ThemeMode(str, Enum):
    light = "light"
    dark  = "dark"

class StyleSettings(BaseModel):
    """
    Pydantic model for theme-based style settings.
    Provides a semantic typographic scale (h1–h6, p), separate font families for headings and paragraphs,
    and chart palettes. Colors and palettes are auto-filled based on `mode`.
    """
    # theme switch
    mode: ThemeMode = ThemeMode.light

    # semantic typographic scale
    font_size_h1: int = 32
    font_size_h2: int = 28
    font_size_h3: int = 24
    font_size_h4: int = 20
    font_size_h5: int = 16
    font_size_h6: int = 14
    font_size_p:  int = 12

    # default font families
    font_family_headings: str = "Montserrat, sans-serif"
    font_family_paragraphs: str = "Lato, Arial, Helvetica, sans-serif"

    # layout
    title_column_width: str = "150px"
    chart_label_font_size: int = 12

    # theme-driven colors (auto-filled)
    main_color:       Optional[str] = Field(None)
    secondary_color:  Optional[str] = Field(None)
    accent_color_1:   Optional[str] = Field(None)
    accent_color_2:   Optional[str] = Field(None)
    heading_color:    Optional[str] = Field(None)
    paragraph_color:  Optional[str] = Field(None)
    background_color: Optional[str] = Field(None)

    # chart color palettes
    chart_palette_sequential:   Optional[List[str]] = Field(None)
    chart_palette_diverging:    Optional[List[str]] = Field(None)
    chart_palette_categorical:  Optional[List[str]] = Field(None)

    @root_validator(pre=True)
    def _fill_theme_defaults(cls, values: Dict) -> Dict:
        palettes = {
            ThemeMode.light: {
                # base colors
                "main_color":       "#003049",
                "secondary_color":  "#b4a269",
                "accent_color_1":   "#aea06c",
                "accent_color_2":   "#aea06c",
                "heading_color":    "#003049",
                "paragraph_color":  "#333333",
                "background_color": "#FFFFFF",
                # chart palettes
                "chart_palette_sequential":   ["#f7fbff","#deebf7","#9ecae1","#3182bd"],
                "chart_palette_diverging":    ["#d7191c","#fdae61","#ffffbf","#abdda4","#2b83ba"],
                "chart_palette_categorical":  ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e"],
            },
            ThemeMode.dark: {
                "main_color":       "#FAFAFA",
                "secondary_color":  "#FFD700",
                "accent_color_1":   "#FFA500",
                "accent_color_2":   "#FF8C00",
                "heading_color":    "#FAFAFA",
                "paragraph_color":  "#EEEEEE",
                "background_color": "#121212",
                "chart_palette_sequential":   ["#2c3e50","#34495e","#4b6584","#78a5a3"],
                "chart_palette_diverging":    ["#b2182b","#f4a582","#f7f7f7","#92c5de","#2166ac"],
                "chart_palette_categorical":  ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"],
            }
        }
        mode = values.get("mode", ThemeMode.light)
        for field, default in palettes.get(mode, {}).items():
            values.setdefault(field, default)
        return values


# ─── instantiate both themes ────────────────────────────────────────────
light_settings: StyleSettings = StyleSettings(mode=ThemeMode.light)
dark_settings:  StyleSettings = StyleSettings(mode=ThemeMode.dark)


def get_theme_settings(mode: ThemeMode) -> StyleSettings:
    """
    Retrieve the global light or dark settings instance.
    """
    return light_settings if mode is ThemeMode.light else dark_settings


def update_settings_from_dict(overrides: dict, mode: ThemeMode) -> None:
    """
    Update either `light_settings` or `dark_settings` in-place from dict `overrides`.

    - `overrides` may include any fields (colors, fonts, layout, palettes).
    - `mode` selects which settings instance to modify.
    """
    # select global instance
    instance = get_theme_settings(mode)
    # merge current values with overrides
    merged = instance.dict()
    merged.update(overrides)
    # create a temporary instance to re-apply root defaults and validation
    temp = StyleSettings(**merged)
    # mutate the existing settings instance so imports remain valid
    for key, value in temp.dict().items():
        setattr(instance, key, value)



class Theme(BaseModel):
    logo_url: Optional[str] = None
    font_family: str = "Helvetica, Arial, sans-serif"
    base_font_size: int = 14
    cover_background_url: Optional[str] = None
    title_color: str = "#000"
    current_date: str = Field(default_factory=lambda: "{{CURRENT_DATE}}")

    def logo_img_html(self, position: str = "slide-logo") -> str:
        return f'<div class="{position}"><img src="{self.logo_url}" alt="logo" crossOrigin="anonymous"></div>' if self.logo_url else ""


class Presentation(BaseModel):
    title: str
    subtitle: Optional[str] = None
    slides: List[Slide]
    theme: Theme = Field(default_factory=Theme)

    def render(self) -> str:
        slides_html = []

        # add the slide template
        self.slides.append(self._slide_template())


        total = len(self.slides)-1 # do not add the final template slide


        slides_html += [s.render(i + 1, total, self.theme) for i, s in enumerate(self.slides)]
        return BASE_TEMPLATE.render(
            title=self.title,
            font_family=self.theme.font_family,
            base_font_size=self.theme.base_font_size,
            title_color=self.theme.title_color,
            slides="".join(slides_html),
        )

    def _slide_template(self)->Slide:

        # 1) Four rows:
        #    - First row “auto” for our split tutorial
        #    - Then the three demo rows (100px, 2fr, 1fr)
        row_definitions = ["auto", "100px", "2fr", "1fr"]

        # 2) Twelve columns mixing px and fr
        col_definitions = [
            "50px", "1fr", "2fr", "100px", "3fr", "1fr",
            "200px", "2fr", "1fr", "150px", "4fr", "1fr"
        ]

        # 3) Tutorial cells in row 1:
        cells: List[GridCell] = [
            # Left tutorial text (cols 1–6) with detailed fr explanation
            GridCell(
                row=1, col=1, col_span=6,
                element=TextElement(
                    text=(
                        "<strong>Tutorial: How fr Units Are Calculated</strong><br><br>"
                        "1. <em>Start with total container height</em> (e.g. 800px).<br>"
                        "2. <em>Subtract auto/fixed rows</em>:<br>"
                        "   • Row 1 (auto) → measured by content, say 200px<br>"
                        "   • Row 2 (fixed) → exactly 100px<br>"
                        "   → Used: 300px<br>"
                        "3. <em>Free space</em> = 800px − 300px = 500px<br>"
                        "4. <em>Total fr shares</em> = 2fr + 1fr = 3 shares<br>"
                        "5. <em>One share</em> = 500px ÷ 3 ≈ 166.67px<br>"
                        "6. <em>Allocate</em>:<br>"
                        "   • Row 3 (2fr) → 2×166.67px ≈ 333.33px<br>"
                        "   • Row 4 (1fr) → 1×166.67px ≈ 166.67px<br><br>"
                        "→ That’s how 2fr can take twice the free space and still leave one share for 1fr!"
                    ),
                    font_size=14,
                    h_align=HorizontalAlign.left,
                    v_align=VerticalAlign.top
                ),
                padding="12px",
                background_color="#f9f9f9"
            ),
            # Right tutorial code (cols 7–12)
            GridCell(
                row=1, col=7, col_span=6,
                element=TextElement(
                    text=(
                        "<pre style=\"font-size:12px; white-space:pre-wrap;\">"
                        "row_defs = ['auto', '100px', '2fr', '1fr']\n"
                        "col_defs = ['50px','1fr','2fr','100px','3fr','1fr',\n"
                        "             '200px','2fr','1fr','150px','4fr','1fr']\n\n"
                        "slide = GridLayout(\n"
                        "    row_definitions=row_defs,\n"
                        "    col_definitions=col_defs,\n"
                        "    gap='10px',\n"
                        "    cells=...  # see demo rows below\n"
                        ")\n"
                        "</pre>"
                    ),
                    font_size=12,
                    h_align=HorizontalAlign.left,
                    v_align=VerticalAlign.top
                ),
                padding="12px",
                background_color="#ffffff"
            )
        ]

        # 4) Demo cells for rows 2–4
        for r in range(2, 5):  # rows 2, 3, 4
            for c in range(1, 13):  # cols 1–12
                label = f"R{r}({row_definitions[r - 1]}), C{c}({col_definitions[c - 1]})"
                cells.append(
                    GridCell(
                        row=r,
                        col=c,
                        element=TextElement(
                            text=label,
                            font_size=12,
                            h_align=HorizontalAlign.center,
                            v_align=VerticalAlign.center
                        )
                    )
                )

        # 5) Build and render the layout
        slide_layout = GridLayout(
            row_definitions=row_definitions,
            col_definitions=col_definitions,
            gap="10px",
            cells=cells,
            width="100%",
            height="100%"
        )

        return Slide(
            title="Slide Template",
            layout=slide_layout,

        )

base_template_str = (
    Path(__file__).parent
    / "base_templates"
    / "base_template.html"
).read_text(encoding="utf-8")
BASE_TEMPLATE = Environment().from_string(base_template_str)