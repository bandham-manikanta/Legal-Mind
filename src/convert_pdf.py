# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: llm692_venv
#     language: python
#     name: python3
# ---

# %%
# Install required packages
# pip install fpdf pygments

from fpdf import FPDF
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import os
import glob

def remove_non_latin1(text):
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def convert_py_to_pdf(python_file, pdf_file=None):
    """Convert a single Python file to PDF"""
    if pdf_file is None:
        pdf_file = os.path.splitext(python_file)[0] + '.pdf'
    
    # Read Python file
    with open(python_file, 'r', encoding='utf-8') as file:
        code = file.read()
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=10)
    
    # Split by lines and add to PDF
    for line in code.split('\n'):
        line = remove_non_latin1(line)
        # Truncate long lines to prevent errors
        if len(line) > 180:  # FPDF has limitations on cell width
            chunks = [line[i:i+180] for i in range(0, len(line), 180)]
            for chunk in chunks:
                pdf.cell(0, 5, txt=chunk, ln=1)
        else:
            pdf.cell(0, 5, txt=line, ln=1)
    
    # Save the PDF
    pdf.output(pdf_file)
    print(f"Converted {python_file} to {pdf_file}")
    return pdf_file

def convert_all_py_files_in_directory():
    """Convert all Python files in the current directory to PDF"""
    # Get all .py files in the current directory
    python_files = glob.glob("*.py")
    
    if not python_files:
        print("No Python files found in the current directory.")
        return
    
    print(f"Found {len(python_files)} Python files to convert.")
    
    # Convert each file
    converted_files = []
    for py_file in python_files:
        try:
            pdf_file = convert_py_to_pdf(py_file)
            converted_files.append(pdf_file)
        except Exception as e:
            print(f"Error converting {py_file}: {e}")
    
    print(f"\nSuccessfully converted {len(converted_files)} files to PDF:")
    for file in converted_files:
        print(f"- {file}")

# Run the conversion
if __name__ == "__main__":
    convert_all_py_files_in_directory()


# %%
