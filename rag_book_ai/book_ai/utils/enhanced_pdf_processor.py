import fitz
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

class EnhancedPDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.document = None
        self.filename = file_path.split('/')[-1]
        self.file_extension = self.filename.split('.')[-1].lower()
        self.page_char_counts = []
        self.logger = logging.getLogger(__name__)

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
        """Extract text with better formatting preservation"""
        full_text = ""
        self.page_char_counts = []
        try:
            if not self.document:
                raise ValueError("Document not open.")
            
            for page_num in range(len(self.document)):
                page = self.document[page_num]
                
                # Extract text with layout preservation
                page_text = page.get_text("text")
                
                # Clean up the text while preserving structure
                page_text = self._clean_page_text(page_text)
                
                full_text += f"\n--- PAGE {page_num + 1} ---\n" + page_text + "\n"
                self.page_char_counts.append(len(page_text))
                
            return full_text
        except Exception as e:
            raise Exception(f"Error extracting text from {self.filename}: {e}")

    def _clean_page_text(self, text: str) -> str:
        """Clean and format page text while preserving structure"""
        # Remove excessive whitespace but preserve line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        return text

    def extract_enhanced_bookmarks(self) -> List[Dict[str, Any]]:
        """Extract bookmarks with enhanced metadata"""
        try:
            if not self.document:
                raise ValueError("Document not open.")
            
            toc = self.document.get_toc()
            if not toc:
                return []
                
            bookmarks = []
            for level, title, page in toc:
                page_idx = max(0, page - 1)  # Convert to 0-based indexing
                
                # Clean and normalize title
                clean_title = self._clean_title(title)
                
                # Determine bookmark type
                bookmark_type = self._classify_bookmark(clean_title, level)
                
                bookmarks.append({
                    'title': clean_title,
                    'original_title': title,
                    'level': level,
                    'page': page_idx,
                    'type': bookmark_type,
                    'sequence': len(bookmarks)
                })
                
            return bookmarks
        except Exception as e:
            raise Exception(f"Error extracting bookmarks from {self.filename}: {e}")

    def _clean_title(self, title: str) -> str:
        """Clean and normalize titles"""
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Remove common artifacts
        title = re.sub(r'^\.+|\.+$', '', title)
        title = re.sub(r'^\d+\.?\s*', '', title)  # Remove leading numbers
        
        return title

    def _classify_bookmark(self, title: str, level: int) -> str:
        """Classify bookmark type based on title and level"""
        title_lower = title.lower()
        
        # Chapter patterns
        if any(keyword in title_lower for keyword in ['chapter', 'chapitre', 'kapittel']):
            return 'chapter'
        
        # Part patterns
        if any(keyword in title_lower for keyword in ['part', 'partie', 'del', 'book']):
            return 'part'
        
        # Section patterns
        if any(keyword in title_lower for keyword in ['section', 'sektion', 'sección']):
            return 'section'
        
        # Appendix patterns
        if any(keyword in title_lower for keyword in ['appendix', 'annex', 'annexe', 'apéndice']):
            return 'appendix'
        
        # Introduction/Conclusion patterns
        if any(keyword in title_lower for keyword in ['introduction', 'preface', 'foreword']):
            return 'introduction'
        
        if any(keyword in title_lower for keyword in ['conclusion', 'summary', 'epilogue']):
            return 'conclusion'
        
        # Level-based classification
        if level == 1:
            return 'main_chapter'
        elif level == 2:
            return 'subsection'
        elif level >= 3:
            return 'subsubsection'
        
        return 'general'

    def identify_enhanced_chapters_regex(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced regex-based chapter identification with better patterns"""
        chapter_matches = []
        
        # Comprehensive patterns for different document types
        patterns = [
            # Chapter patterns (Level 1)
            {
                'pattern': r'(?:^|\n)\s*(?:Chapter|CHAPTER|Ch\.?)\s+(\d+|[IVX]+)(?:\s*[:\-–—]\s*|\s+)([^\n]+)',
                'level': 1,
                'type': 'chapter'
            },
            {
                'pattern': r'(?:^|\n)\s*(\d+)\.\s+([A-Z][A-Za-z\s,\-–—]{3,50})(?=\n|$)',
                'level': 1,
                'type': 'numbered_section'
            },
            
            # Part patterns (Level 0)
            {
                'pattern': r'(?:^|\n)\s*(?:Part|PART|Book|BOOK)\s+(\d+|[IVX]+)(?:\s*[:\-–—]\s*|\s+)([^\n]+)',
                'level': 0,
                'type': 'part'
            },
            
            # Section patterns (Level 2)
            {
                'pattern': r'(?:^|\n)\s*(\d+\.\d+)(?:\s*[:\-–—]\s*|\s+)([A-Z][^\n]+)',
                'level': 2,
                'type': 'section'
            },
            {
                'pattern': r'(?:^|\n)\s*(?:Section|SECTION)\s+(\d+\.\d+|\d+)(?:\s*[:\-–—]\s*|\s+)([^\n]+)',
                'level': 2,
                'type': 'section'
            },
            
            # Subsection patterns (Level 3)
            {
                'pattern': r'(?:^|\n)\s*(\d+\.\d+\.\d+)(?:\s*[:\-–—]\s*|\s+)([A-Z][^\n]+)',
                'level': 3,
                'type': 'subsection'
            },
            
            # Letter-based patterns
            {
                'pattern': r'(?:^|\n)\s*([A-Z])\.\s+([A-Z][^\n]+)',
                'level': 2,
                'type': 'lettered_section'
            },
            {
                'pattern': r'(?:^|\n)\s*([a-z])\)\s+([A-Z][^\n]+)',
                'level': 3,
                'type': 'lettered_subsection'
            },
            
            # Special sections
            {
                'pattern': r'(?:^|\n)\s*(Introduction|INTRODUCTION|Preface|PREFACE|Foreword|FOREWORD)(?:\s*[:\-–—]\s*|\s*)([^\n]*)',
                'level': 1,
                'type': 'introduction'
            },
            {
                'pattern': r'(?:^|\n)\s*(Conclusion|CONCLUSION|Summary|SUMMARY|Epilogue|EPILOGUE)(?:\s*[:\-–—]\s*|\s*)([^\n]*)',
                'level': 1,
                'type': 'conclusion'
            },
            {
                'pattern': r'(?:^|\n)\s*(Appendix|APPENDIX|Annex|ANNEX)\s*([A-Z]?)(?:\s*[:\-–—]\s*|\s+)([^\n]*)',
                'level': 1,
                'type': 'appendix'
            }
        ]
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            level = pattern_info['level']
            section_type = pattern_info['type']
            
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                title_parts = [part for part in match.groups() if part and part.strip()]
                
                if len(title_parts) >= 2:
                    # Number/identifier and title
                    identifier = title_parts[0].strip()
                    title = title_parts[1].strip()
                    full_title = f"{identifier} {title}" if identifier.replace('.', '').isdigit() or identifier in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' else title
                elif len(title_parts) == 1:
                    # Just title
                    full_title = title_parts[0].strip()
                else:
                    continue
                
                # Clean the title
                full_title = self._clean_title(full_title)
                
                if len(full_title) > 3:  # Filter out very short titles
                    chapter_matches.append({
                        'title': full_title,
                        'level': level,
                        'type': section_type,
                        'position': match.start(),
                        'match_text': match.group(0).strip(),
                        'identifier': title_parts[0].strip() if title_parts else None
                    })
        
        # Sort by position in document
        chapter_matches.sort(key=lambda x: x['position'])
        
        # Remove duplicates and very similar entries
        filtered_matches = self._filter_duplicate_chapters(chapter_matches)
        
        return filtered_matches

    def _filter_duplicate_chapters(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out duplicate or very similar chapter entries"""
        filtered = []
        
        for chapter in chapters:
            is_duplicate = False
            
            for existing in filtered:
                # Check if titles are very similar (>80% similarity)
                if self._calculate_similarity(chapter['title'], existing['title']) > 0.8:
                    # Keep the one with better formatting or more information
                    if len(chapter['title']) > len(existing['title']):
                        filtered.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(chapter)
        
        return filtered

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        str1_words = set(str1.lower().split())
        str2_words = set(str2.lower().split())
        
        if not str1_words or not str2_words:
            return 0.0
        
        intersection = str1_words.intersection(str2_words)
        union = str1_words.union(str2_words)
        
        return len(intersection) / len(union)

    def find_page_number(self, char_index: int) -> int:
        """Find page number for a given character index"""
        current_index = 0
        for page_num, char_count in enumerate(self.page_char_counts):
            if current_index <= char_index < current_index + char_count:
                return page_num
            current_index += char_count
        return len(self.page_char_counts) - 1

    def extract_content_for_chapter(self, start_page: int, end_page: int) -> str:
        """Extract content for a specific chapter"""
        content = ""
        try:
            for page_num in range(start_page, min(end_page + 1, len(self.document))):
                page = self.document[page_num]
                page_text = page.get_text("text")
                content += self._clean_page_text(page_text) + "\n\n"
        except Exception as e:
            self.logger.warning(f"Error extracting content for pages {start_page}-{end_page}: {e}")
        
        return content.strip()

    def build_enhanced_toc(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build enhanced table of contents with proper hierarchy"""
        enhanced_toc = []
        
        for i, chapter in enumerate(chapters):
            # Calculate end page
            if i + 1 < len(chapters):
                end_page = max(0, chapters[i + 1]['page'] - 1)
            else:
                end_page = len(self.document) - 1 if self.document else 0
            
            # Extract content
            content = self.extract_content_for_chapter(chapter['page'], end_page)
            
            # Create enhanced chapter entry
            enhanced_chapter = {
                'id': i + 1,
                'title': chapter['title'],
                'level': chapter['level'],
                'type': chapter.get('type', 'general'),
                'start_page': chapter['page'],
                'end_page': end_page,
                'content': content,
                'word_count': len(content.split()) if content else 0,
                'sequence': i + 1,
                'parent_id': None,
                'children': []
            }
            
            enhanced_toc.append(enhanced_chapter)
        
        # Build hierarchy
        enhanced_toc = self._build_hierarchy(enhanced_toc)
        
        return enhanced_toc

    def _build_hierarchy(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build hierarchical structure for chapters"""
        for i, chapter in enumerate(chapters):
            # Find parent (previous chapter with lower level)
            for j in range(i - 1, -1, -1):
                if chapters[j]['level'] < chapter['level']:
                    chapter['parent_id'] = chapters[j]['id']
                    chapters[j]['children'].append(chapter['id'])
                    break
        
        return chapters

    def process_pdf(self) -> Optional[Dict[str, Any]]:
        """Main processing method with enhanced TOC extraction"""
        if not self.open_document():
            return None

        try:
            # Extract text
            full_text = self.extract_text_fitz()
            if not full_text:
                raise Exception(f"Could not extract text from {self.filename}.")

            # Try bookmark extraction first
            bookmarks = self.extract_enhanced_bookmarks()
            
            if bookmarks:
                self.logger.info(f"Found {len(bookmarks)} bookmarks in PDF")
                chapters = bookmarks
            else:
                self.logger.info("No bookmarks found, using regex extraction")
                chapters = self.identify_enhanced_chapters_regex(full_text)
            
            # Build enhanced TOC
            enhanced_toc = self.build_enhanced_toc(chapters)
            
            # Generate summary statistics
            stats = self._generate_stats(enhanced_toc)
            
            return {
                "filename": self.filename,
                "page_count": len(self.document),
                "hierarchical_chunks": enhanced_toc,
                "chunks": enhanced_toc,  # For backward compatibility
                "bookmarks": bookmarks,
                "extraction_method": "bookmarks" if bookmarks else "regex",
                "stats": stats,
                "full_text": full_text[:1000] + "..." if len(full_text) > 1000 else full_text  # Sample for debugging
            }
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise
        finally:
            self.close_document()

    def _generate_stats(self, toc: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the extracted TOC"""
        stats = {
            'total_chapters': len(toc),
            'levels_found': list(set(chapter['level'] for chapter in toc)),
            'types_found': list(set(chapter.get('type', 'general') for chapter in toc)),
            'avg_chapter_length': sum(chapter['word_count'] for chapter in toc) / len(toc) if toc else 0,
            'total_words': sum(chapter['word_count'] for chapter in toc)
        }
        
        # Count by level
        stats['by_level'] = {}
        for chapter in toc:
            level = chapter['level']
            if level not in stats['by_level']:
                stats['by_level'][level] = 0
            stats['by_level'][level] += 1
        
        return stats

# Backward compatibility - extend the original class
class PDFProcessor(EnhancedPDFProcessor):
    """Enhanced PDF processor with backward compatibility"""
    pass
