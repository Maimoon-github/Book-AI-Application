import fitz
import re
from typing import List, Dict, Any, Optional

class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.document = None
        self.filename = file_path.split('/')[-1]
        self.file_extension = self.filename.split('.')[-1].lower()
        self.page_char_counts = []

    def open_document(self) -> bool:
        try:
            if self.file_extension == "pdf":
                self.document = fitz.open(self.file_path)
                return True
            else:
                raise ValueError(f"Unsupported file type: {self.file_extension}")
        except Exception as e:
            raise Exception(f"Error opening {self.filename}: {e}")

    def close_document(self) -> None:
        if self.document:
            self.document.close()
            self.document = None

    def extract_text_fitz(self) -> str:
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
            raise Exception(f"Error extracting text from {self.filename}: {e}")

    def extract_bookmarks_fitz(self) -> List[tuple]:
        try:
            if not self.document:
                raise ValueError("Document not open.")
            
            toc = self.document.get_toc()
            if not toc:
                return []
                
            bookmarks = []
            for level, title, page in toc:
                page_idx = page - 1 if page > 0 else 0
                bookmarks.append((title, level, page_idx))
            return bookmarks
        except Exception as e:
            raise Exception(f"Error extracting bookmarks from {self.filename}: {e}")

    def identify_chapters_regex(self, text: str) -> List[tuple]:
        chapter_starts = []
        patterns = [
            (1, r"(?:^|\n)\s*Chapter\s+(\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (1, r"(?:^|\n)\s*Part\s+(\d+|[IVX]+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (1, r"(?:^|\n)\s*Section\s+(\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (1, r"(?:^|\n)\s*\d+\.\s+([A-Z][^\n]+)"),
            (2, r"(?:^|\n)\s*(\d+\.\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (3, r"(?:^|\n)\s*(\d+\.\d+\.\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (2, r"(?:^|\n)\s*([A-Z]\.)\s+([^\n]+)"),
            (3, r"(?:^|\n)\s*([a-z]\.)\s+([^\n]+)")
        ]
        
        for level, pattern in patterns:
            for match in re.finditer(pattern, text):
                title = match.group(0).strip()
                chapter_starts.append((title, level, match.start()))
        
        chapter_starts.sort(key=lambda x: x[2])
        return chapter_starts

    def find_page_number(self, char_index: int) -> int:
        current_index = 0
        for page_num, char_count in enumerate(self.page_char_counts):
            if current_index <= char_index < current_index + char_count:
                return page_num
            current_index += char_count
        return len(self.page_char_counts) - 1

    def process_bookmark_chunks(self, bookmarks: List[tuple]) -> List[Dict[str, Any]]:
        chunks = []
        sorted_bookmarks = sorted(bookmarks, key=lambda x: x[2])
        
        for i, (title, level, start_page) in enumerate(sorted_bookmarks):
            end_page = self.document.page_count - 1
            for j in range(i+1, len(sorted_bookmarks)):
                next_title, next_level, next_page = sorted_bookmarks[j]
                if next_level <= level:
                    end_page = next_page - 1
                    break
            
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

    def process_regex_chunks(self, chapter_starts: List[tuple], full_text: str) -> List[Dict[str, Any]]:
        chunks = []
        
        for i, (title, level, start_pos) in enumerate(chapter_starts):
            start_page = self.find_page_number(start_pos)
            
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
        chunks.sort(key=lambda x: (x["start_page"], x["level"]))
        
        hierarchical_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["id"] = i
            chunk_copy["children"] = []
            hierarchical_chunks.append(chunk_copy)
        
        for i, chunk in enumerate(hierarchical_chunks):
            parent_id = None
            for j in range(i-1, -1, -1):
                if hierarchical_chunks[j]["level"] < chunk["level"]:
                    parent_id = hierarchical_chunks[j]["id"]
                    hierarchical_chunks[j]["children"].append(i)
                    break
            chunk["parent_id"] = parent_id
        
        return hierarchical_chunks

    def process_pdf(self) -> Optional[Dict[str, Any]]:
        if not self.open_document():
            return None

        try:
            full_text = self.extract_text_fitz()
            if not full_text:
                raise Exception(f"Could not extract text from {self.filename}.")

            # Extract and process content
            bookmarks = self.extract_bookmarks_fitz()
            
            if bookmarks:
                chunks = self.process_bookmark_chunks(bookmarks)
            else:
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
