
from .base import BasePydanticModel, BaseObjectOrm, TDAG_ENDPOINT
from .utils import (is_process_running, get_network_ip,
                    TDAG_CONSTANTS,
                    DATE_FORMAT, AuthLoaders, make_request, set_types_in_table, request_to_datetime, serialize_to_json, bios_uuid)
from typing import Optional, List,Annotated, Union
from pydantic import constr, Field,conint
import datetime


# Define a reusable HexColor annotated type
HexColor = Annotated[
    str,
    Field(
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="HEX color in format #RRGGBB",
        min_length=7,
        max_length=7,
    )
]


class Theme(BasePydanticModel,BaseObjectOrm):
    id: int
    theme_type: constr(max_length=15) = Field(
        ..., description="‘standard’ or ‘custom’"
    )
    name: constr(max_length=100)
    created: datetime.datetime
    updated: datetime.datetime

    mode: constr(max_length=5) = Field(
        ..., description="‘light’ or ‘dark’"
    )
    editor_background: Optional[int] = Field(
        None, description="FK to Background.id"
    )

    font_family_headings: Optional[constr(max_length=100)] = Field(
        "", description="--font-family-headings"
    )
    font_family_paragraphs: Optional[constr(max_length=100)] = Field(
        "", description="--font-family-paragraphs"
    )

    font_size_h1: conint(ge=1) = Field(48, description="--font-size-h1 (px)")
    font_size_h2: conint(ge=1) = Field(40, description="--font-size-h2 (px)")
    font_size_h3: conint(ge=1) = Field(32, description="--font-size-h3 (px)")
    font_size_h4: conint(ge=1) = Field(24, description="--font-size-h4 (px)")
    font_size_h5: conint(ge=1) = Field(20, description="--font-size-h5 (px)")
    font_size_h6: conint(ge=1) = Field(16, description="--font-size-h6 (px)")
    font_size_p:  conint(ge=1) = Field(14, description="--font-size-p  (px)")

    primary_color:          HexColor = Field("#0d6efd")
    secondary_color:      HexColor = Field("#6c757d")
    accent_color_1:         HexColor= Field("#198754")
    accent_color_2:         HexColor = Field("#ffc107")
    heading_color:           HexColor = Field("#212529")
    paragraph_color:         HexColor= Field("#495057")
    light_paragraph_color:   HexColor = Field("#6c757d")
    background_color:       HexColor = Field("#ffffff")

    title_column_width:     constr(max_length=15) = Field(
        "150px", description="--title-column-width"
    )
    chart_label_font_size:  conint(ge=1) = Field(
        12, description="--chart-label-font-size (px)"
    )

    chart_palette_sequential:   List[HexColor] = Field(
        default_factory=list,
        description="List of 5 HEX colours for sequential palettes",
    )
    chart_palette_diverging:    List[HexColor] = Field(
        default_factory=list,
        description="List of 5 HEX colours for diverging palettes",
    )
    chart_palette_categorical:  List[HexColor] = Field(
        default_factory=list,
        description="List of 6 HEX colours for categorical palettes",
    )


class Folder(BasePydanticModel, BaseObjectOrm):
    id:Optional[int]=None
    name:str
    slug:str


    @classmethod
    def get_or_create(cls, *args, **kwargs):
        url = f"{cls.get_object_url()}/get-or-create/"
        payload = {"json": kwargs}
        r = make_request(
            s=cls.build_session(),
            loaders=cls.LOADERS,
            r_type="POST",
            url=url,
            payload=payload
        )
        if r.status_code not in [200, 201]:
            raise Exception(f"Error appending creating: {r.text}")
        # Return a new instance of AssetCategory built from the response JSON.
        return cls(**r.json())

class Presentation(BasePydanticModel, BaseObjectOrm):
    id: int
    folder: int
    title: str = Field(..., max_length=255)
    description: Optional[str] = Field("", description="Optional text")
    created_at: datetime.datetime
    updated_at:  datetime.datetime
    theme: Union[int,Theme]

    @classmethod
    def get_or_create_by_title(cls, *args, **kwargs):
        url = f"{cls.get_object_url()}/get_or_create_by_title/"
        payload = {"json": kwargs}
        r = make_request(
            s=cls.build_session(),
            loaders=cls.LOADERS,
            r_type="POST",
            url=url,
            payload=payload
        )
        if r.status_code not in [200, 201]:
            raise Exception(f"Error appending creating: {r.text}")
        # Return a new instance of AssetCategory built from the response JSON.
        return cls(**r.json())