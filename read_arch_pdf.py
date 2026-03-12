from pypdf import PdfReader

pdf_path = r"c:\Users\yaweh\OneDrive\Documents\Medumba AI\Technical Architecture.pdf"
reader = PdfReader(pdf_path)
print(f"pages {len(reader.pages)}")

for index, page in enumerate(reader.pages):
    text = (page.extract_text() or "").replace("\n", " ")
    print(f"---PAGE {index + 1}---")
    print(text[:2500])
