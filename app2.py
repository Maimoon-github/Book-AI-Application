# import os
# import shutil
# import streamlit as st
# from pathlib import Path

# def setup_temp_directory(temp_dir_name="temp_uploads"):
#     """Sets up a temporary directory for uploaded files."""
#     current_script_dir = Path(__file__).parent
#     temp_upload_dir = current_script_dir / temp_dir_name
#     temp_upload_dir.mkdir(parents=True, exist_ok=True)
#     return temp_upload_dir

# def main():
#     st.title("Document Upload Tool")
#     st.write("Upload a document for processing")
    
#     # Set up temp directory
#     temp_uploads_directory = setup_temp_directory()
    
#     # Define allowed file types
#     allowed_extensions = [".pdf", ".epub", ".mobi", ".docx", ".txt"]
    
#     # Create a file uploader widget
#     uploaded_file = st.file_uploader(
#         "Choose a document", 
#         type=[ext.replace(".", "") for ext in allowed_extensions]
#     )
    
#     if uploaded_file is not None:
#         # Save the uploaded file to the temporary directory
#         try:
#             file_path = Path(uploaded_file.name)
#             destination_path = temp_uploads_directory / file_path.name
            
#             with open(destination_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             st.success(f"Document '{file_path.name}' uploaded successfully!")
#             st.write(f"Stored temporarily at: {destination_path}")
            
#             # You can add further processing here
#             st.write("Ready for processing!")
            
#             # Optional: Add a button to clear the temp files
#             if st.button("Clear temporary files"):
#                 if destination_path.exists():
#                     os.remove(destination_path)
#                     st.write(f"Removed temporary file: {destination_path}")
        
#         except Exception as e:
#             st.error(f"An error occurred during file upload: {e}")

# if __name__ == "__main__":
#     main()


import os
import shutil
import streamlit as st
from pathlib import Path
import fitz  # Import PyMuPDF


def setup_temp_directory(temp_dir_name="temp_uploads"):
    """
    Sets up a temporary directory for uploaded files.
    Creates the directory if it doesn't exist.
    Returns the path to the temporary directory.
    """
    current_script_dir = Path(__file__).parent
    temp_upload_dir = current_script_dir / temp_dir_name
    temp_upload_dir.mkdir(parents=True, exist_ok=True)
    return temp_upload_dir


def extract_pdf_toc(pdf_path):
    """
    Extracts the table of contents (bookmarks) from a PDF.
    Args:
        pdf_path (Path): The path to the PDF file.
    Returns:
        list: A list of dictionaries, where each dictionary represents a bookmark
              with 'level', 'title', and 'page' keys.  Returns an empty list if
              no TOC is found or if an error occurs.
    """
    bookmarks = []
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        if toc:
            for item in toc:
                level, title, page = item[0], item[1], item[2]
                bookmarks.append({"level": level, "title": title, "page": page})
        doc.close()
    except Exception as e:
        st.error(f"Error extracting TOC from PDF: {e}")
        return []
    return bookmarks


def display_toc(bookmarks):
    """
    Displays the table of contents in a Streamlit app.
    Args:
        bookmarks (list): A list of bookmark dictionaries.
    """
    if bookmarks:
        st.subheader("Table of Contents:")
        for item in bookmarks:
            level, title, page = item['level'], item['title'], item['page']
            st.markdown(f"{'  ' * (level - 1)}- **{title}** (Page: {page})")
    else:
        st.info("No Table of Contents found in this PDF.")


def main():
    st.title("PDF Table of Contents Extractor")
    st.write("Upload a PDF document to extract its Table of Contents.")

    # Set up temp directory
    temp_uploads_directory = setup_temp_directory()

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    uploaded_file_path = None

    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name
            destination_path = temp_uploads_directory / file_name

            with open(destination_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File '{file_name}' uploaded successfully!")
            st.info(f"Stored temporarily at: `{destination_path}`")
            uploaded_file_path = destination_path

        except Exception as e:
            st.error(f"An error occurred during file upload: {e}")

    # --- TOC Extraction and Display ---
    if uploaded_file_path and uploaded_file_path.suffix.lower() == ".pdf":
        st.header("Table of Contents Extraction")
        with st.spinner("Extracting Table of Contents..."):
            extracted_bookmarks = extract_pdf_toc(uploaded_file_path)

        display_toc(extracted_bookmarks)

    # --- Cleanup ---
    if uploaded_file_path and st.button("Clear temporary file"):
        if uploaded_file_path.exists():
            os.remove(uploaded_file_path)
            st.write(f"Removed temporary file: {uploaded_file_path}")
            st.rerun()


if __name__ == "__main__":
    main()



