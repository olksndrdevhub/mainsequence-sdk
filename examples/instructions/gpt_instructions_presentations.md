LLM Guide: How to Build Presentations with mainsequence.client
This document provides a complete guide for programmatically creating presentations. Follow these steps precisely to generate presentations that include Plotly charts.

Core Concepts
Presentation: The main container for your slides. It has a title, description, and theme.

Theme: Controls the visual appearance (colors, fonts) of the presentation and its charts.

Slide: A single page within the presentation. Its content is defined by a structured JSON body.

Artifact: A file stored in Main Sequence, such as an HTML file containing a chart. Slides embed these artifacts to display visual content.

Prerequisites (Assumed to be True)
Authentication: Your environment is already authenticated, and the mainsequence.client (ms_client) is available.

Chart Data: You have access to the data required to generate the necessary charts.

Target Location: You know the name of the Folder where the final presentation should be saved.

Workflow: From Chart to Presentation
Follow this four-step process. The order is critical: you must set the theme before creating any charts to ensure they are styled correctly.

Step 1: Create or Select the Presentation and Set the Theme
First, set up the presentation and apply the visual theme. This step ensures that any subsequent Plotly charts will automatically adopt the correct styling.

Get Folder: Retrieve the folder where the presentation will be saved.

Get or Create Presentation: Use Presentation.get_or_create_by_title() to avoid duplicates.

Get and Apply Theme: Retrieve a Theme by name, apply it to the presentation using patch(), and then call presentation.theme.set_plotly_theme(). This method must be called before you create any figures.

Step 2: Generate a Plotly Chart as an HTML Artifact
All charts must be created with Plotly. After setting the theme, create your chart and convert it to an HTML artifact.

Create the Chart: Generate your figure using plotly.express or plotly.graph_objects. Because the theme was set in Step 1, this figure will automatically have the correct styling.

Export to HTML String: Convert the Plotly figure to an HTML string using the to_html() method with the following required parameters:

full_html=False: To create a partial HTML snippet.

include_plotlyjs="cdn": To load the Plotly library from a CDN.

config={"responsive": True, "displayModeBar": False}: For responsive charts without the mode bar.

Upload HTML as an Artifact: Use the helper function below to write the HTML string to a temporary file and upload it using Artifact.upload_file(). This returns the artifact_id.

import mainsequence.client as ms_client
import plotly.express as px
import tempfile
import os

def create_chart_artifact(fig, artifact_name: str, bucket_name: str = "PresentationAssets") -> str:
    """
    Converts a Plotly figure to an HTML string and uploads it as an Artifact.

    Args:
        fig: The Plotly figure object.
        artifact_name: A unique name for the artifact.
        bucket_name: The storage bucket for the artifact.

    Returns:
        The ID of the created artifact.
    """
    # Export to HTML with required settings
    chart_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False}
    )

    # Write to a temporary file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
        tmp.write(chart_html)
        temp_path = tmp.name

    try:
        artifact = ms_client.Artifact.upload_file(
            filepath=temp_path,
            name=artifact_name,
            created_by_resource_name="LLMPresentationGenerator", # Or other identifier
            bucket_name=bucket_name,
        )
        print(f"Created artifact '{artifact_name}' with ID: {artifact.id}")
        return artifact.id
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

Step 3: Add a Slide and Populate its Content
Add a New Slide: Call presentation.add_slide() to append a new, empty slide.

Construct the Body: The slide's content is a JSON object. Use the TextParagraph and AppNode helpers to build it with a heading, the embedded chart artifact, and a descriptive paragraph.

Patch the Slide: Use the slide.patch(body=...) method to update the slide with the generated content.

Complete Example Script
This script demonstrates the entire workflow in the correct order.

import mainsequence.client as ms_client
import json
import pandas as pd
import plotly.express as px
import tempfile
import os

# --- Helper Function (from Step 2) ---
def create_chart_artifact(fig, artifact_name: str, bucket_name: str = "PresentationAssets") -> str:
    chart_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False}
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
        tmp.write(chart_html)
        temp_path = tmp.name
    try:
        artifact = ms_client.Artifact.upload_file(
            filepath=temp_path, name=artifact_name,
            created_by_resource_name="LLMPresentationGenerator", bucket_name=bucket_name
        )
        return artifact.id
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Main Generation Logic ---

# 1. Define configuration
PRESENTATION_TITLE = "LLM-Generated Sales Report"
FOLDER_NAME = "Automated Reports"
THEME_NAME = "Main Sequence"

# 2. Set up presentation and theme FIRST (Step 1)
folder = ms_client.Folder.get_or_create(name=FOLDER_NAME)
presentation = ms_client.Presentation.get_or_create_by_title(
    title=PRESENTATION_TITLE,
    folder=folder.id,
    description="This presentation was generated automatically."
)
try:
    theme = ms_client.Theme.get(name=THEME_NAME)
    presentation.patch(theme_id=theme.id)
    # CRITICAL: Set the theme before creating any figures
    presentation.theme.set_plotly_theme()
    print(f"Applied theme '{THEME_NAME}' to Plotly.")
except ms_client.client.ResourceNotFound:
    print(f"Theme '{THEME_NAME}' not found. Using default.")

# 3. Now, create the Plotly chart (Step 2)
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 title="Iris Dataset: Sepal Length vs. Width")

# 4. Create the chart artifact
chart_artifact_id = create_chart_artifact(fig, artifact_name="iris_scatter_plot")

# 5. Add and populate a slide (Step 3)
new_slide = presentation.add_slide()

slide_content_body = {
    "type": "doc",
    "content": [
        ms_client.TextParagraph.heading("Analysis of Iris Sepal Dimensions", level=2).model_dump(),
        json.loads(ms_client.AppNode.make_app_node(id=chart_artifact_id).model_dump_json()),
        ms_client.TextParagraph.paragraph(
            "This chart displays the correlation between sepal width and length across different species."
        ).model_dump(),
    ],
}

new_slide.patch(body=json.dumps(slide_content_body))

print(f"\nSuccessfully created presentation '{PRESENTATION_TITLE}' with ID: {presentation.id}")
print(f"Added and populated new slide with ID: {new_slide.id}")
