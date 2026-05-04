import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

if __name__ == "__main__":
    pdf_path = r"d:\Project\NLP\ML\Criminal Law\The-Indian-Evidence-Act-1872.pdf"
    if os.path.exists(pdf_path):
        text = extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters from {pdf_path}")
        print("First 500 characters:")
        print(text[:500])
    else:
        print(f"File not found: {pdf_path}")
