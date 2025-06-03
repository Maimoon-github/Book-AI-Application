import streamlit as st
import tempfile
import os
import PyPDF2
import fitz  # PyMuPDF

def extract_pdf_content(file_path):
    content = ""
    bookmarks = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # Extract text from each page
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                content += page_text + "\n"
        # Try to extract bookmarks (if available)
        try:
            outlines = reader.outline
            bookmarks = parse_outlines(outlines, reader)
        except Exception as e:
            st.warning(f"Bookmarks extraction not supported or encountered an error: {e}")
    return content, bookmarks

def parse_outlines(outlines, reader, parent_title=""):
    result = []
    for item in outlines:
        if isinstance(item, list):
            result.extend(parse_outlines(item, reader, parent_title))
        else:
            title = item.title if hasattr(item, "title") else "Untitled"
            if parent_title:
                title = f"{parent_title} > {title}"
            page_number = None
            try:
                page_number = reader.get_destination_page_number(item)
            except Exception:
                pass
            result.append((title, page_number))
    return result

def extract_pdf_content_with_structure(file_path):
    doc = fitz.open(file_path)
    content = ""
    structural_elements = []  # list to store detected headings as (text, page number, font size, font name)
    
    # Process each page in the document
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") == 0:  # this is a text block
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        fontsize = span["size"]
                        fontname = span["font"]
                        content += text + " "
                        
                        # Using a naive heuristic to detect headings/chapter titles:
                        # If the font size is larger than 14 or the font indicates bold style
                        if fontsize > 14 or "Bold" in fontname or "bold" in fontname:
                            structural_elements.append((text, page_idx + 1, fontsize, fontname))
                    content += "\n"
    return content, structural_elements

st.title("Document Uploader")

uploaded_file = st.file_uploader(
    "Choose a document file",
    type=["pdf", "epub", "mobi", "docx", "txt"]
)

if uploaded_file is not None:
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, prefix="uploaded_", suffix=ext) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    st.success(f"File '{filename}' saved to: {temp_path}")

    if filename.lower().endswith(".pdf"):
        # PyPDF2 extraction
        text, bookmarks = extract_pdf_content(temp_path)
        st.subheader("Extracted Text (PyPDF2)")
        st.text_area("PDF Content", text, height=300)
        st.subheader("Extracted Bookmarks/Table of Contents")
        if bookmarks:
            for title, page in bookmarks:
                st.write(f"{title} (Page: {page + 1 if page is not None else 'N/A'})")
        else:
            st.write("No bookmarks found.")

        # fitz (PyMuPDF) extraction
        text_fitz, headings = extract_pdf_content_with_structure(temp_path)
        st.subheader("Extracted Text (PyMuPDF)")
        st.text_area("PDF Content (PyMuPDF)", text_fitz, height=300)
        st.subheader("Inferred Headings/Chapters (PyMuPDF)")
        if headings:
            for h in headings:
                st.write(f"Heading: {h[0]} | Page: {h[1]} | Font Size: {h[2]} | Font: {h[3]}")
        else:
            st.write("No headings detected by PyMuPDF.")

