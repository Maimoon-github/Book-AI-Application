import streamlit as st
import os
import re
import tempfile
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import fitz  # PyMuPDF

class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.document = None
        self.filename = os.path.basename(file_path)
        self.file_extension = os.path.splitext(self.filename)[1].lower()
        self.page_char_counts = []  # Store character count per page for faster lookups

    def open_document(self) -> bool:
        """Opens the PDF document using PyMuPDF (fitz)."""
        try:
            if self.file_extension == ".pdf":
                self.document = fitz.open(self.file_path)
                return True
            else:
                raise ValueError(f"Unsupported file type: {self.file_extension}")
        except Exception as e:
            st.error(f"Error opening {self.filename}: {e}")
            return False

    def close_document(self) -> None:
        """Closes the PDF document."""
        if self.document:
            self.document.close()
            self.document = None

    def extract_text_fitz(self) -> str:
        """Extracts text from PDF using PyMuPDF (fitz)."""
        full_text = ""
        self.page_char_counts = []
        try:
            if not self.document:
                raise ValueError("Document not open.")
            for page in self.document:
                page_text = page.get_text()
                full_text += page_text
                self.page_char_counts.append(len(page_text))
            return full_text
        except Exception as e:
            st.error(f"Error extracting text from {self.filename}: {e}")
            return ""

    def extract_bookmarks_fitz(self) -> List[Tuple[str, int, int]]:
        """Extracts bookmarks using PyMuPDF (fitz)."""
        try:
            if not self.document:
                raise ValueError("Document not open.")
            
            toc = self.document.get_toc()
            if not toc:
                return []
                
            bookmarks = []
            for level, title, page in toc:
                # PyMuPDF uses 1-indexed page numbers, convert to 0-indexed
                page_idx = page - 1 if page > 0 else 0
                bookmarks.append((title, level, page_idx))
            return bookmarks
        except Exception as e:
            st.error(f"Error extracting bookmarks from {self.filename}: {e}")
            return []

    def identify_chapters_regex(self, text: str) -> List[Tuple[str, int, int]]:
        """Identifies chapters using improved regular expressions and infers level."""
        chapter_starts = []
        patterns = [
            (1, r"(?:^|\n)\s*Chapter\s+(\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # Chapter 1: Title or Chapter 1 Title
            (1, r"(?:^|\n)\s*Part\s+(\d+|[IVX]+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # Part I: Title or Part I Title
            (1, r"(?:^|\n)\s*Section\s+(\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # Section 1: Title
            (1, r"(?:^|\n)\s*\d+\.\s+([A-Z][^\n]+)"),  # 1. TITLE FORMAT
            (2, r"(?:^|\n)\s*(\d+\.\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # 1.1: Title or 1.1 Title
            (3, r"(?:^|\n)\s*(\d+\.\d+\.\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # 1.1.1: Title or 1.1.1 Title
            (2, r"(?:^|\n)\s*([A-Z]\.)\s+([^\n]+)"),  # A. Subtitle format
            (3, r"(?:^|\n)\s*([a-z]\.)\s+([^\n]+)")   # a. Sub-subtitle format
        ]
        
        for level, pattern in patterns:
            for match in re.finditer(pattern, text):
                title = match.group(0).strip()
                chapter_starts.append((title, level, match.start()))
        
        # Sort by position in text
        chapter_starts.sort(key=lambda x: x[2])
        return chapter_starts

    def find_page_number(self, char_index: int) -> int:
        """Finds the page number corresponding to a character index based on page character counts."""
        current_index = 0
        for page_num, char_count in enumerate(self.page_char_counts):
            if current_index <= char_index < current_index + char_count:
                return page_num
            current_index += char_count
        return self.document.page_count - 1  # Return last page if not found

    def process_bookmark_chunks(self, bookmarks: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
        """Process PDF using bookmarks to create content chunks."""
        chunks = []
        
        # Sort bookmarks by page number
        sorted_bookmarks = sorted(bookmarks, key=lambda x: x[2])
        
        for i, (title, level, start_page) in enumerate(sorted_bookmarks):
            # Determine end page - either before next bookmark at same/lower level or document end
            end_page = self.document.page_count - 1
            for j in range(i+1, len(sorted_bookmarks)):
                next_title, next_level, next_page = sorted_bookmarks[j]
                if next_level <= level:  # Same level or higher in hierarchy
                    end_page = next_page - 1
                    break
            
            # Extract content from the pages
            content = ""
            for pg in range(start_page, min(end_page + 1, self.document.page_count)):
                content += self.document.load_page(pg).get_text() + "\n"
            
            chunks.append({
                "title": title,
                "level": level,
                "start_page": start_page,
                "end_page": end_page,
                "content": content.strip()
            })
        
        return chunks

    def process_regex_chunks(self, chapter_starts: List[Tuple[str, int, int]], full_text: str) -> List[Dict[str, Any]]:
        """Process PDF using regex-identified chapters to create content chunks."""
        chunks = []
        
        for i, (title, level, start_pos) in enumerate(chapter_starts):
            # Find start page for this chunk
            start_page = self.find_page_number(start_pos)
            
            # Find end position and page
            if i + 1 < len(chapter_starts):
                end_pos = chapter_starts[i + 1][2]
                end_page = self.find_page_number(end_pos - 1)
            else:
                end_pos = len(full_text)
                end_page = self.document.page_count - 1
            
            content = full_text[start_pos:end_pos].strip()
            chunks.append({
                "title": title,
                "level": level,
                "start_page": start_page,
                "end_page": end_page,
                "content": content
            })
        
        return chunks

    def build_hierarchical_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize chunks into a hierarchical structure with parent-child relationships."""
        # Sort by level and then by start_page to ensure proper ordering
        chunks.sort(key=lambda x: (x["start_page"], x["level"]))
        
        # Add parent_id and children fields
        hierarchical_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["id"] = i
            chunk_copy["children"] = []
            hierarchical_chunks.append(chunk_copy)
        
        # Build parent-child relationships
        for i, chunk in enumerate(hierarchical_chunks):
            parent_id = None
            # Find the nearest previous chunk with a lower level
            for j in range(i-1, -1, -1):
                if hierarchical_chunks[j]["level"] < chunk["level"]:
                    parent_id = hierarchical_chunks[j]["id"]
                    hierarchical_chunks[j]["children"].append(i)
                    break
            chunk["parent_id"] = parent_id
        
        return hierarchical_chunks

    def process_pdf(self) -> Optional[Dict[str, Any]]:
        """Main function to process the PDF."""
        if not self.open_document():
            return None

        try:
            with st.spinner("Extracting text..."):
                full_text = self.extract_text_fitz()
                if not full_text:
                    st.error(f"Could not extract text from {self.filename}.")
                    return None

            with st.spinner("Processing document structure..."):
                # Extract Bookmarks (Table of Contents)
                bookmarks = self.extract_bookmarks_fitz()
                
                # Process based on bookmarks or regex
                if bookmarks:
                    st.info(f"Bookmarks found in {self.filename}. Using bookmarks for chapter identification.")
                    chunks = self.process_bookmark_chunks(bookmarks)
                else:
                    st.info(f"No bookmarks found in {self.filename}. Using pattern detection for chapter identification.")
                    chapter_starts = self.identify_chapters_regex(full_text)
                    chunks = self.process_regex_chunks(chapter_starts, full_text)
                
                # Build hierarchical structure
                hierarchical_chunks = self.build_hierarchical_chunks(chunks)
                
            return {
                "filename": self.filename,
                "chunks": chunks,
                "hierarchical_chunks": hierarchical_chunks,
                "bookmarks": bookmarks,
                "page_count": self.document.page_count
            }
        finally:
            self.close_document()

def display_hierarchical_toc(hierarchical_chunks):
    """Display a hierarchical table of contents without nested expanders."""
    # Extract all chunks and sort by page number and level
    sorted_chunks = sorted(hierarchical_chunks, key=lambda x: (x["start_page"], x["level"]))
    
    # Display TOC as a hierarchical list
    st.write("### Document Structure")
    
    # Create a dictionary for level-based indentation display
    for chunk in sorted_chunks:
        # Calculate the actual display level by counting parents
        display_level = 0
        parent_id = chunk["parent_id"]
        while parent_id is not None:
            display_level += 1
            parent = next((c for c in hierarchical_chunks if c["id"] == parent_id), None)
            if parent:
                parent_id = parent["parent_id"]
            else:
                parent_id = None
        
        # Create indentation with HTML spaces
        indent = "&nbsp;" * (display_level * 4)
        
        # Display the chunk with proper indentation
        st.markdown(f"{indent}📄 **{chunk['title']}** (Pages {chunk['start_page']+1}-{chunk['end_page']+1})", unsafe_allow_html=True)
    
    # Create a section for viewing chunk content
    st.write("---")
    st.subheader("View Chapter Content")
    
    # Create options for the dropdown with formatted titles
    chunk_options = [f"{c['title']} (Pages {c['start_page']+1}-{c['end_page']+1})" for c in sorted_chunks]
    selected_chunk_title = st.selectbox("Select a chapter to view:", chunk_options)
    
    # Find the selected chunk
    selected_index = chunk_options.index(selected_chunk_title)
    selected_chunk = sorted_chunks[selected_index]
    
    # Display the content of the selected chunk
    with st.expander(f"Content of {selected_chunk['title']}", expanded=True):
        st.text_area("Chapter text", selected_chunk['content'], height=300)

def export_chunks_to_csv(chunks, filename):
    """Export chunks to CSV file."""
    df = pd.DataFrame(chunks)
    # Select only the columns we want to export
    export_df = df[['title', 'level', 'start_page', 'end_page']]
    # Add 1 to page numbers for display (convert from 0-indexed to 1-indexed)
    export_df['start_page'] += 1
    export_df['end_page'] += 1
    # Create a downloadable CSV
    return export_df.to_csv(index=False).encode('utf-8')

def main():
    st.title("📚 Book AI Processor")
    st.write("Upload a PDF to extract its structure and content by chapters and subtopics.")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Structure", "Search", "Export"])
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name
        
        try:
            with st.spinner("Processing PDF..."):
                processor = PDFProcessor(temp_file_path)
                result = processor.process_pdf()
            
            if result:
                with tab1:
                    st.success(f"✅ Processing complete: {result['filename']} ({result['page_count']} pages)")
                    
                    # Display hierarchical table of contents
                    st.subheader("Document Structure")
                    display_hierarchical_toc(result["hierarchical_chunks"])
                
                with tab2:
                    st.subheader("Search Document")
                    search_query = st.text_input("Enter search term:")
                    if search_query:
                        st.write(f"Searching for: '{search_query}'")
                        results = []
                        for chunk in result["chunks"]:
                            if search_query.lower() in chunk["content"].lower():
                                results.append(chunk)
                        
                        if results:
                            st.write(f"Found {len(results)} matches:")
                            for r in results:
                                with st.expander(f"{r['title']} (Pages {r['start_page']+1}-{r['end_page']+1})"):
                                    # Highlight matches in content preview
                                    content_preview = r["content"][:500]
                                    st.markdown(content_preview.replace(search_query, f"**{search_query}**"), unsafe_allow_html=True)
                        else:
                            st.info("No matches found.")
                
                with tab3:
                    st.subheader("Export Options")
                    csv = export_chunks_to_csv(result["chunks"], result["filename"])
                    st.download_button(
                        label="Download Structure as CSV",
                        data=csv,
                        file_name=f"{os.path.splitext(result['filename'])[0]}_structure.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Processing failed.")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {e}")

if __name__ == "__main__":
    main()