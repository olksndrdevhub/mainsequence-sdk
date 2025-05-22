from typing import Dict, Callable, List, Optional, Union, Any

from pandas import DataFrame
from pydantic import BaseModel, conlist, PositiveFloat, Field, validator

import plotly.express as px

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


class PieChartArgs(BaseModel):
    labels: conlist(str, min_length=1)
    values: conlist(float, min_length=1) # Allow any float, not just PositiveFloat
    height: int = 350
    width: str = "100%"
    title: Optional[str] = None # Optional title for the chart

@register_function("pie_chart")
def pie_chart(args: PieChartArgs) -> str:
    """Render an interactive Plotly pie chart."""
    fig = px.pie(
        names=args.labels,
        values=args.values,
        height=args.height,
        title=args.title
    )
    # return html fragment only, to be embedded
    return fig.to_html(include_plotlyjs=False, full_html=False)


class BarChartArgs(BaseModel):
    x: conlist(str, min_length=1)
    y: conlist(float, min_length=1) # Allow any float
    orientation: str = "v"
    height: int = 350
    width: str = "100%"
    title: Optional[str] = None # Optional title for the chart
    labels: Optional[Dict[str, str]] = None # e.g. {"x": "X-axis Label", "y": "Y-axis Label"}

@register_function("bar_chart")
def bar_chart(args: BarChartArgs) -> str:
    """Render an interactive Plotly bar chart."""
    fig = px.bar(
        x=args.x,
        y=args.y,
        orientation=args.orientation,
        height=args.height,
        title=args.title,
        labels=args.labels
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


class LineChartArgs(BaseModel):
    x: conlist(Any, min_length=1) # X-axis values (can be numbers, dates, or strings)
    y: conlist(float, min_length=1) # Y-axis values
    series_name: Optional[str] = None # Name for the line/series, used in legend
    height: int = 350
    width: str = "100%"
    title: Optional[str] = None # Optional title for the chart
    labels: Optional[Dict[str, str]] = None # e.g. {"x": "X-axis Label", "y": "Y-axis Label"}
    markers: bool = False # Whether to show markers on the line points

@register_function("line_chart")
def line_chart(args: LineChartArgs) -> str:
    """Render an interactive Plotly line chart."""
    # Plotly Express works best if x and y are part of a dataframe or dict
    # For a single line, we can pass them directly.
    # If multiple lines were needed in one chart, data structure would be more complex.
    fig = px.line(
        x=args.x,
        y=args.y,
        height=args.height,
        title=args.title,
        labels=args.labels,
        markers=args.markers
    )
    if args.series_name:
        fig.update_traces(name=args.series_name, showlegend=True)

    return fig.to_html(include_plotlyjs=False, full_html=False)

class DataTableArgs(BaseModel):
    headers: conlist(str, min_length=1)
    rows: List[conlist(Any, min_length=1)] # List of rows, each row is a list of cell values
    title: Optional[str] = None
    table_class: str = "table table-striped table-sm" # Bootstrap classes by default
    max_height: Optional[str] = None # e.g., "400px" for scrollable table

    @validator("rows")
    def _validate_row_lengths(cls, rows, values):
        if "headers" in values and values["headers"]:
            header_len = len(values["headers"])
            for i, row in enumerate(rows):
                if len(row) != header_len:
                    raise ValueError(f"Row {i+1} has {len(row)} cells, but header has {header_len} columns.")
        return rows

@register_function("data_table")
def data_table(args: DataTableArgs) -> str:
    """Render an HTML table from headers and rows of data."""
    html = ""
    if args.title:
        html += f"<h4>{args.title}</h4>"

    style = f"max-height: {args.max_height}; overflow-y: auto;" if args.max_height else ""
    html += f'<div style="{style}">'
    html += f'<table class="{args.table_class}">'

    html += "<thead><tr>"
    for header in args.headers:
        html += f"<th>{header}</th>"
    html += "</tr></thead>"

    html += "<tbody>"
    for row in args.rows:
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</tbody>"
    html += "</table>"
    html += "</div>"
    return html

# Update annotations for Pydantic model discovery in model.py
bar_chart.__annotations__["return_model"] = BarChartArgs
pie_chart.__annotations__["return_model"] = PieChartArgs
line_chart.__annotations__["return_model"] = LineChartArgs
data_table.__annotations__["return_model"] = DataTableArgs
