import streamlit as st
import tempfile
import os
import fitz  # PyMuPDF

# ——— Full‐document + heading extractor ———
def extract_full_with_structure(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    headings = []
    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    full_text += text + " "
                    size, font = span["size"], span["font"]
                    if size > 14 or "Bold" in font:
                        headings.append((text, page_idx+1, size, font))
                full_text += "\n"
    return full_text, headings

# ——— Table‐only extractor (naïve) ———
def extract_tables_only(file_path):
    doc = fitz.open(file_path)
    tables = []
    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        # get words with positions
        words = page.get_text("words")  # x0, y0, x1, y1, word, block_no, line_no, word_no
        # group words by line_no
        lines = {}
        for x0, y0, x1, y1, w, bno, lno, wno in words:
            lines.setdefault((bno, lno), []).append((x0, w))
        # pick lines with > 3 columns as “table rows”
        for (bno, lno), items in lines.items():
            if len(items) > 3:
                row = " | ".join([w for _, w in sorted(items, key=lambda x: x[0])])
                tables.append((page_idx+1, row))
    return tables

# ——— Streamlit app ———
st.title("Document Uploader")

uploaded_file = st.file_uploader(
    "Choose a document file",
    type=["pdf", "epub", "mobi", "docx", "txt"]
)

if uploaded_file:
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    st.success(f"Saved to: {temp_path}")

    if filename.lower().endswith(".pdf"):
        mode = st.radio("Extraction mode", ["Full Text + Headings", "Table Content Only"])

        if mode == "Full Text + Headings":
            try:
                text, headings = extract_full_with_structure(temp_path)
                st.subheader("Full Text")
                st.text_area("PDF Content", text, height=300)
                st.subheader("Detected Headings")
                if headings:
                    for t, p, sz, f in headings:
                        st.write(f"{t} (Page {p}) – size:{sz} font:{f}")
                else:
                    st.write("No headings found.")
            except Exception as e:
                st.error(f"Full‐document extraction failed: {e}")

        else:  # Table content only
            try:
                tables = extract_tables_only(temp_path)
                st.subheader("Extracted Table Rows")
                if tables:
                    for p, row in tables:
                        st.write(f"Page {p}: {row}")
                else:
                    st.write("No table‐like rows detected.")
            except Exception as e:
                st.error(f"Table extraction failed: {e}")
    else:
        st.warning("Non‐PDF support not implemented yet.")
