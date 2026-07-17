import markdown
from xhtml2pdf import pisa
import sys

def convert_md_to_pdf(md_file, pdf_file):
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
            
        # Weasyprint/xhtml2pdf usually struggles with raw LaTeX math. 
        # But we will convert the markdown to HTML, and let xhtml2pdf render it.
        html = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
        
        # Add basic styling
        html_styled = f"""
        <html>
        <head>
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12pt;
                    line-height: 1.5;
                    color: #333333;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                }}
                pre {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    font-family: Courier, monospace;
                    font-size: 10pt;
                    white-space: pre-wrap;
                }}
                code {{
                    background-color: #f8f9fa;
                    font-family: Courier, monospace;
                    padding: 2px 4px;
                    border-radius: 3px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #dee2e6;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f8f9fa;
                }}
                .math {{
                    font-family: "Times New Roman", Times, serif;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        with open(pdf_file, "wb") as result_file:
            pisa_status = pisa.CreatePDF(
                html_styled,
                dest=result_file
            )
            
        if pisa_status.err:
            print("Error generating PDF:", pisa_status.err)
            sys.exit(1)
        else:
            print(f"Successfully generated {pdf_file}")
            sys.exit(0)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    convert_md_to_pdf(sys.argv[1], sys.argv[2])
