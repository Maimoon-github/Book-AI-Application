import streamlit as st
import tempfile
import os

st.title("Document Uploader")

# Allow user to upload a file (PDF, EPUB, MOBI, DOCX, TXT)
uploaded_file = st.file_uploader(
    "Choose a document file",
    type=["pdf", "epub", "mobi", "docx", "txt"]
)

if uploaded_file is not None:
    # Get file extension
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1]
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, prefix="uploaded_", suffix=ext) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    st.success(f"File '{filename}' saved to: {temp_path}")