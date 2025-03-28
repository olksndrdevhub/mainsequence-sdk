#!/usr/bin/env python3


from pathlib import Path


import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# Optional: for PDF generation using pdfkit (requires wkhtmltopdf installed)
from weasyprint import HTML

def generate_report():
    """
    Renders a static HTML report from Jinja2 templates and (optionally) saves it as PDF.
    """

    # 1. Setup the Jinja2 environment:
    #
    #    - Point it to the directory that holds your base.html and report.html files.
    #    - autoescape=False is often acceptable for simple text-based reporting;
    #      if you’re processing user input, consider enabling an autoescape policy.
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)

    # 2. Load the derived template (which extends base.html):
    template = env.get_template('report.html')

    # 3. Prepare the data context for your template blocks.
    #
    #    For demonstration, we’ll pretend “mcook” is your data source for logos,
    #    authors, etc. We’ll also embed a snippet of text from your PDF to show
    #    how you might incorporate real content.

    current_date_str = datetime.now().strftime('%Y-%m-%d')

    template_context = {
        "current_date": current_date_str,
        "report_id": "MC-2025-XYZ",
        "authors": "Main Sequence AI ",
        "sector": "Global Equities",
        "region": "International",
        "topics": [
            "Diversification",
            "Equities",
            "Interest Rates",
            "Technology",
        ],
        "current_year": datetime.now().year,

        # Example placeholders. Replace them with your real data:
        "report_title": "Global Strategy Views: Diversify to Amplify",
        "summary": (
            "We are entering a more benign part of the cycle with continued economic growth "
            "and easing policy rates. This environment has historically been supportive of "
            "equities, but it also highlights the importance of broader diversification."
        ),
        "pdf_path": "weasy_output_report.pdf",
        "report_content": (
            """
            <h2>Overview</h2>
            <p>
                Longer-term rates are likely to remain elevated, given rising government deficits
                and term premia. Nevertheless, a lower probability of a near-term recession
                suggests opportunities for positive returns in various sectors, including
                technology and selective value areas like financials.
            </p>
            <p>
                This shifting landscape underscores the value of broadening our focus—beyond
                just the US or just mega-cap tech—into regional equities, mid-caps, and
                so-called “Ex-Tech Compounders.” The approach aims to enhance risk-adjusted
                returns as growth rates converge across regions.
            </p>
            <h2>Key Takeaways</h2>
            <ul>
                <li>
                    <strong>Diversify to amplify returns:</strong> The next cycle may require a broader
                    portfolio to capture alpha from multiple regions and factors.
                </li>
                <li>
                    <strong>Technology remains pivotal:</strong> But with a rising need for physical
                    infrastructure to power AI and data centers, it creates new beneficiaries across
                    older industrial and electrification themes.
                </li>
                <li>
                    <strong>Rates on diverging paths:</strong> Central banks have shifted to easing, yet
                    bond yields remain 'higher for longer,' implying a cap on equity valuation
                    expansions.
                </li>
            </ul>
            """
        )
    }

    # 4. Render the template to a string:
    rendered_html = template.render(template_context)

    # 5. Write the rendered HTML to a file (e.g. "output_report.html"):
    output_html_path = os.path.join(os.path.dirname(__file__), 'output_report.html')
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    print(f"HTML report generated: {output_html_path}")

    # 6. Optional: Convert the HTML to PDF using pdfkit (requires wkhtmltopdf installed).
    #    Uncomment the lines below if you want to generate a PDF in the same directory.
    #
    #    If pdfkit or wkhtmltopdf is not installed on your system, install them:
    #       pip install pdfkit
    #       (and install wkhtmltopdf from your OS package manager or from https://wkhtmltopdf.org/)
    #
    pdf_output_path = os.path.join(os.path.dirname(__file__), 'weasy_output_report.pdf')
    base_path = Path(__file__).resolve().parent

    HTML(string=rendered_html, base_url=str(base_path)).write_pdf(pdf_output_path)

    print("Generated HTML:", output_html_path)
    print("Generated PDF:", pdf_output_path)

if __name__ == "__main__":
    generate_report()
