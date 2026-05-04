import PyPDF2
import os

def search_in_pdf(pdf_path, search_term):
    found = False
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if search_term in text:
                print(f"Found '{search_term}' on page {i+1}")
                print("Snippet:")
                start_idx = text.find(search_term)
                print(text[max(0, start_idx-200):min(len(text), start_idx+500)])
                found = True
                # We don't break so we can see all occurrences
    if not found:
        print(f"'{search_term}' not found in {pdf_path}")

if __name__ == "__main__":
    pdf_path = r"d:\Project\NLP\ML\Criminal Law\Indian Peal Code.pdf"
    if os.path.exists(pdf_path):
        search_in_pdf(pdf_path, "302")
    else:
        print(f"File not found: {pdf_path}")
