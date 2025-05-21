import plotly.express as px
from model import register_function
from pydantic import BaseModel, conlist, PositiveFloat

class PieChartArgs(BaseModel):
    labels: conlist(str, min_items=1)
    values: conlist(PositiveFloat, min_items=1)
    height: int = 350
    width: str = "100%"

@register_function("pie_chart")
def pie_chart(args: PieChartArgs) -> str:
    """Render an interactive Plotly pie chart."""
    fig = px.pie(
        names=args.labels,
        values=args.values,
        height=args.height
    )
    # return html fragment only, to be embedded
    return fig.to_html(include_plotlyjs=False, full_html=False)


class BarChartArgs(BaseModel):
    x: conlist(str, min_items=1)
    y: conlist(PositiveFloat, min_items=1)
    orientation: str = "v"
    height: int = 350
    width: str = "100%"

@register_function("bar_chart")
def bar_chart(args: BarChartArgs) -> str:
    """Render an interactive Plotly bar chart."""
    fig = px.bar(
        x=args.x,
        y=args.y,
        orientation=args.orientation,
        height=args.height
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)

bar_chart.__annotations__["return_model"] = BarChartArgs
pie_chart.__annotations__["return_model"] = PieChartArgs
