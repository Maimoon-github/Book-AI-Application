import streamlit as st
import tempfile
import os
import PyPDF2

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
        text, bookmarks = extract_pdf_content(temp_path)
        st.subheader("Extracted Text")
        st.text_area("PDF Content", text, height=300)
        st.subheader("Extracted Bookmarks/Table of Contents")
        if bookmarks:
            for title, page in bookmarks:
                st.write(f"{title} (Page: {page + 1 if page is not None else 'N/A'})")
        else:
            st.write("No bookmarks found.")