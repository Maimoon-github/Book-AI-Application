# Standard libraries first
import sys
import warnings
import re
import math
import uuid
from collections import Counter
import json
import os
import io

# Import dependency manager
from .dependencies import dependency_manager

# Initialize dependency manager and check dependencies
dependency_manager.check_all()

# Import libraries based on availability
if dependency_manager.is_available("PyPDF2"):
    import PyPDF2
else:
    PyPDF2 = None
    warnings.warn("PyPDF2 not found. PDF processing will be disabled.")

# Import packages based on dependency availability
if dependency_manager.is_available("nltk"):
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    try:
        # Try to ensure we have the required NLTK data
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading required NLTK data...", file=sys.stderr)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
else:
    warnings.warn("NLTK not found. Basic tokenization will be used.")
    def sent_tokenize(text):
        # Basic sentence tokenization fallback
        return [s.strip() + '.' for s in text.split('.') if s.strip()]
    stopwords = type('', (), {'words': lambda *args: []})()

# Import numpy if available
if dependency_manager.is_available("numpy"):
    import numpy as np
else:
    warnings.warn("numpy not found. Some features will be disabled.")
    np = None

# Import torch if available
if dependency_manager.is_available("torch"):
    import torch
else:
    warnings.warn("torch not found. Some features will be disabled.")

# Import sentence-transformers if available
if dependency_manager.is_available("sentence_transformers"):
    from sentence_transformers import SentenceTransformer, util
else:
    warnings.warn("sentence-transformers not found. Semantic search will be disabled.")
    SentenceTransformer = None
    # Create dummy util object
    util = type('', (), {
        'cos_sim': lambda *args: 0.0,
    })()

# Import FAISS if available
if dependency_manager.is_available("faiss"):
    import faiss
else:
    warnings.warn("faiss not found. Fast similarity search will be disabled.")
    faiss = None

# Import BM25 if available
if dependency_manager.is_available("rank_bm25"):
    from rank_bm25 import BM25Okapi
else:
    warnings.warn("rank-bm25 not found. Keyword search will be degraded.")
    BM25Okapi = None

# Let the dependency manager handle the status printing
class HierarchicalChunker:
    """
    Splits document content into semantic chunks based on chapters and context.
    """
    def __init__(self):
        """Initialize the chunker."""
        self.fallback_max_chars = 4000  # For basic chunking when no chapter structure
    
    def process_document(self, file_obj):
        """
        Process a document file and return semantic chunks.
        
        Args:
            file_obj: File-like object to process
            
        Returns:
            list: List of chunks with text content and metadata
        """
        # Extract text based on file type
        content = self.extract_text_from_pdf(file_obj)
        
        # Try to find chapters
        chapters = self.extract_chapters(content)
        
        if not chapters:
            # No chapters found, fall back to basic chunking
            return self.basic_chunk(content)
              # Process each chapter into smaller semantic chunks
        chunks = []
        for chapter in chapters:
            chapter_chunks = self.chunk_chapter(chapter)
            chunks.extend(chapter_chunks)
            
        return chunks

    def extract_text_from_pdf(self, file):
        """Extract text content from a PDF file with robust error handling."""
        if not dependency_manager.is_available("PyPDF2"):
            raise ImportError("PyPDF2 is required for PDF processing but is not installed. Please run: pip install PyPDF2")

        text = ""
        try:
            # Ensure the file is at the start
            if hasattr(file, 'seek'):
                file.seek(0)

            # Try to detect file corruption before processing
            try:
                header = file.read(5)
                file.seek(0)  # Reset position after reading header
                if not header.startswith(b'%PDF-'):
                    raise ValueError("File does not appear to be a valid PDF (invalid header)")
            except Exception as e:
                print(f"Warning: Could not check PDF header: {e}", file=sys.stderr)

            # Create PDF reader with robust error handling
            try:
                pdf_reader = PyPDF2.PdfReader(file)
            except Exception as e:
                error_msg = str(e).lower()
                if "xref" in error_msg or "startxref" in error_msg:
                    raise ValueError("The PDF file appears to be corrupted (invalid cross-reference table)")
                elif "decrypt" in error_msg:
                    raise ValueError("This PDF may be encrypted. Please remove password protection.")
                elif "eof" in error_msg:
                    raise ValueError("The PDF file is incomplete or corrupted (missing EOF marker)")
                else:
                    raise ValueError(f"Could not open PDF file: {str(e)}")
            
            # Check integrity
            try:
                if not pdf_reader.pages or not hasattr(pdf_reader, 'pages'):
                    raise ValueError("Could not access PDF pages. The file may be corrupted.")
                
                # Get total pages
                total_pages = len(pdf_reader.pages)
                if total_pages == 0:
                    raise ValueError("PDF file appears to be empty (no pages found)")
            except Exception as e:
                raise ValueError(f"Could not read PDF structure: {str(e)}")

            # Extract text from each page with error handling
            empty_pages = 0
            extraction_errors = []

            for page_num in range(total_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    try:
                        page_text = page.extract_text()
                    except AttributeError:
                        # Fallback for some PDF versions
                        page_text = ""
                        for obj in page.get_contents():
                            if hasattr(obj, 'get_data'):
                                page_text += obj.get_data().decode('utf-8', errors='ignore')
                    
                    # Track empty pages for better error messages
                    if not page_text.strip():
                        empty_pages += 1
                        extraction_errors.append(f"Page {page_num + 1}: No text content extracted")
                        continue
                        
                    text += page_text + "\n"
                    
                except Exception as e:
                    extraction_errors.append(f"Page {page_num + 1}: {str(e)}")
                    print(f"Warning: Error extracting text from page {page_num + 1}: {e}", file=sys.stderr)
                    continue  # Continue with next page even if one fails
            
            # Check extracted text quality
            if not text.strip():
                error_details = "\n".join(extraction_errors[:5])  # Show first 5 errors
                if empty_pages == total_pages:
                    raise ValueError(
                        "Could not extract any text from the PDF. The file appears to be scanned or contains images only.\n"
                        f"Details from first few pages:\n{error_details}"
                    )
                else:
                    raise ValueError(
                        "Could not extract readable text from any pages. The file may use an unsupported font or encoding.\n"
                        f"Details from first few pages:\n{error_details}"
                    )
            elif empty_pages > 0:
                print(
                    f"Warning: {empty_pages} out of {total_pages} pages were empty or unreadable.\n"
                    f"First few issues:\n{chr(10).join(extraction_errors[:3])}",
                    file=sys.stderr
                )
            
            return text.strip()

        except Exception as e:
            error_msg = str(e).lower()
            
            # Give more specific error messages
            if "xref" in error_msg or "startxref" in error_msg:
                raise ValueError("The PDF file appears to be corrupted (invalid cross-reference table)")
            elif "eof" in error_msg:
                raise ValueError("The PDF file is incomplete or corrupted (missing EOF marker)")
            elif "codec" in error_msg or "decode" in error_msg:
                raise ValueError("Could not decode the text in this PDF. It may use an unsupported encoding.")
            elif "pdf header" in error_msg:
                raise ValueError("The file does not appear to be a valid PDF")
            elif isinstance(e, ValueError):
                # Pass through our own ValueError messages
                raise
            else:
                print(f"Unexpected error reading PDF: {error_msg}", file=sys.stderr)
                raise ValueError(f"Could not extract text from the PDF: {str(e)}")

    def extract_chapters(self, content):
        """
        Extracts chapters from the content using various patterns.
        Returns list of chapter data or empty list if no chapters found.
        """
        if not content:
            return []
            
        chapter_patterns = [
            # Standard chapter headers
            r'(?:Chapter|CHAPTER)\s+(\d+)[\s:-]+\s*([^\n]+)',
            # Numbered sections
            r'(?m)^\s*(\d+)\.\s+([^\n]+)',
            # Roman numeral chapters
            r'(?:Chapter|CHAPTER)\s+([IVXLC]+)[\s:-]+\s*([^\n]+)',
        ]
        
        for pattern in chapter_patterns:
            chapter_matches = list(re.finditer(pattern, content))
            if chapter_matches:
                chapters = []
                for i, match in enumerate(chapter_matches):
                    try:
                        if match.group(1).isdigit():
                            chapter_num = int(match.group(1))
                        else:
                            # Convert Roman numerals if needed
                            chapter_num = i + 1
                    except:
                        chapter_num = i + 1
                        
                    chapter_title = match.group(2).strip()
                    start_pos = match.start()
                    end_pos = chapter_matches[i+1].start() if i < len(chapter_matches) - 1 else len(content)
                    chapter_text = content[start_pos:end_pos].strip()
                    
                    if len(chapter_text) > 100:  # Ignore very short chapters
                        chapters.append({
                            "number": chapter_num,
                            "title": chapter_title,
                            "text": chapter_text
                        })
                
                if chapters:  # Only return if we found valid chapters
                    return chapters
        
        return []  # No chapters found with any pattern

    def basic_chunk(self, content, max_chars=None):
        """
        Basic chunking fallback when no chapter structure is found.
        Tries to split on paragraph boundaries while respecting max length.
        """
        if not content:
            return []
            
        max_chars = max_chars or self.fallback_max_chars
        chunks = []
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_metadata = {
            "type": "content",
            "section": "main"
        }
        
        for para in paragraphs:
            # If paragraph alone is too long, force split it
            if len(para) > max_chars:
                if current_chunk:
                    chunks.append({
                        "text_content": current_chunk,
                        "metadata": current_metadata.copy()
                    })
                    current_chunk = ""
                
                # Split long paragraph on sentence boundaries
                sentences = sent_tokenize(para)
                sent_chunk = ""
                
                for sent in sentences:
                    if len(sent_chunk) + len(sent) <= max_chars:
                        sent_chunk += sent + " "
                    else:
                        if sent_chunk:
                            chunks.append({
                                "text_content": sent_chunk.strip(),
                                "metadata": current_metadata.copy()
                            })
                        sent_chunk = sent + " "
                
                if sent_chunk:
                    current_chunk = sent_chunk
                
            # Normal paragraph handling
            elif len(current_chunk) + len(para) <= max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "text_content": current_chunk.strip(),
                        "metadata": current_metadata.copy()
                    })
                current_chunk = para + "\n\n"
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text_content": current_chunk.strip(),
                "metadata": current_metadata.copy()
            })
        
        return chunks

    def chunk_chapter(self, chapter_data):
        """
        Split a chapter into semantic chunks based on content and structure.
        """
        if not chapter_data or not chapter_data.get('text'):
            return []
            
        chunks = []
        chapter_num = chapter_data['number']
        chapter_title = chapter_data['title']
        text = chapter_data['text']
        
        # Add chapter header as its own chunk
        chunks.append({
            "text_content": f"Chapter {chapter_num}: {chapter_title}",
            "metadata": {
                "type": "chapter_title",
                "chapter": chapter_num,
                "title": chapter_title
            }
        })
        
        # Split chapter text into semantic chunks
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_metadata = {
            "type": "chapter_content",
            "chapter": chapter_num,
            "chapter_title": chapter_title
        }
        
        for para in paragraphs:
            # Skip the title we already added
            if para == f"Chapter {chapter_num}: {chapter_title}":
                continue
                
            # If adding this paragraph would make chunk too long, save current and start new
            if len(current_chunk) + len(para) > self.fallback_max_chars:
                if current_chunk:
                    chunks.append({
                        "text_content": current_chunk.strip(),
                        "metadata": current_metadata.copy()
                    })
                current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text_content": current_chunk.strip(),
                "metadata": current_metadata.copy()
            })
        
        return chunks

class MetadataEnricher:
    """
    Enriches chunks with detailed metadata for better retrieval.
    """
    def __init__(self):
        """Initialize the metadata enricher with default configurations."""
        # Try to use natural language toolkit for better metadata extraction
        self.has_nltk = dependency_manager.is_available("nltk")
        self.has_transformers = dependency_manager.is_available("sentence_transformers")
        
        # Load stopwords for keyword extraction if available
        if self.has_nltk:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set()
                print("Warning: NLTK stopwords not available", file=sys.stderr)
    
    def enrich_chunks(self, chunks):
        """
        Enriches chunks with additional metadata.
        
        Args:
            chunks (list): List of chunk dictionaries with text_content and basic metadata
            
        Returns:
            list: Same chunks with enhanced metadata
        """
        if not chunks:
            return []
            
        # Make a copy to avoid modifying the input directly
        enriched_chunks = []
        
        # Calculate document statistics for TF-IDF if nltk is available
        word_idf = {}
        if self.has_nltk:
            word_idf = self._calculate_word_idf(chunks)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Create a copy of the chunk to avoid modifying the original
            enriched_chunk = {
                "text_content": chunk["text_content"],
                "metadata": chunk["metadata"].copy() if "metadata" in chunk else {}
            }
            
            # Generate a unique ID for the chunk if it doesn't have one
            if "chunk_id" not in enriched_chunk["metadata"]:
                enriched_chunk["metadata"]["chunk_id"] = str(uuid.uuid4())
                
            # Extract keywords using TF-IDF or simple word frequency
            if self.has_nltk:
                keywords = self._extract_keywords_tfidf(
                    enriched_chunk["text_content"], 
                    word_idf
                )
                enriched_chunk["metadata"]["keywords"] = keywords
            
            # Analyze difficulty level
            difficulty = self._analyze_difficulty(enriched_chunk["text_content"])
            enriched_chunk["metadata"]["difficulty_level"] = difficulty
            
            # Add learning objective template based on available metadata
            chapter_title = enriched_chunk["metadata"].get("chapter_title", "")
            if chapter_title:
                enriched_chunk["metadata"]["learning_objective"] = f"Understand concepts related to {chapter_title}"
            else:
                enriched_chunk["metadata"]["learning_objective"] = "Understand the concepts in this section"
            
            # Process timestamp and add datetime information if not present
            if "creation_date" not in enriched_chunk["metadata"]:
                from datetime import datetime
                enriched_chunk["metadata"]["creation_date"] = datetime.now().strftime("%Y-%m-%d")
            
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def _calculate_word_idf(self, chunks):
        """Calculate inverse document frequency for words across all chunks."""
        from collections import Counter
        import math
        import re
        
        # Dictionary to store all words for TF-IDF calculation
        all_words = Counter()
        chunk_word_counts = []
        
        # First pass: gather word counts from all chunks
        for chunk in chunks:
            text = chunk["text_content"]
            words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) 
                    if w.lower() not in self.stop_words]
            chunk_words = Counter(words)
            all_words.update(chunk_words)
            chunk_word_counts.append(chunk_words)
        
        # Calculate IDF
        num_chunks = len(chunks)
        word_idf = {}
        for word, count in all_words.items():
            # Count how many chunks contain this word
            chunks_with_word = sum(1 for chunk_words in chunk_word_counts if word in chunk_words)
            # IDF formula: log(total chunks / chunks containing word)
            word_idf[word] = math.log(num_chunks / (1 + chunks_with_word))
            
        return word_idf
    
    def _extract_keywords_tfidf(self, text, word_idf, max_keywords=5):
        """Extract keywords using TF-IDF scoring."""
        from collections import Counter
        import re
        
        # Extract and count words
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) 
                if w.lower() not in self.stop_words]
        word_counts = Counter(words)
        
        # Calculate TF-IDF scores
        word_scores = {word: count * word_idf.get(word, 0) for word, count in word_counts.items()}
        
        # Sort by score and extract top keywords
        keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        return [word for word, _ in keywords]
    
    def _analyze_difficulty(self, text):
        """Estimate difficulty level based on text complexity."""
        # Simple heuristic: analyze average word length and sentence length
        if not text.strip():
            return "intermediate"  # Default if text is empty
            
        # Split into words and sentences
        words = text.split()
        if not words:
            return "beginner"
            
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        
        # Split into sentences if nltk is available, otherwise approximate
        if self.has_nltk:
            sentences = sent_tokenize(text)
        else:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
        if not sentences:
            return "beginner"
            
        # Calculate average sentence length
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / max(1, len(sentences))
        
        # Simple difficulty scoring based on word and sentence complexity
        if avg_word_length < 4.5 and avg_sentence_length < 12:
            return "beginner"
        elif avg_word_length < 5.5 and avg_sentence_length < 20:
            return "intermediate"
        else:
            return "advanced"

class HybridRetriever:
    """
    Hybrid retrieval system combining semantic search and keyword-based search.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", top_k=10, rrf_k=60):
        """
        Initialize the hybrid retriever with semantic and keyword components.
        
        Args:
            model_name (str): Name of the sentence transformer model
            top_k (int): Number of results to retrieve from each method
            rrf_k (int): Constant for reciprocal rank fusion scoring
        """
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.chunk_objects = []
        self.vector_index = None
        self.bm25 = None
        self.chunk_ids = []
        
        # Check if we can use semantic search
        self.has_semantic = dependency_manager.is_available("sentence_transformers")
        self.has_faiss = dependency_manager.is_available("faiss")
        self.has_bm25 = dependency_manager.is_available("rank_bm25")
        
        # Initialize embedding model if available
        if self.has_semantic:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                print(f"Initialized hybrid retriever with model: {model_name}", file=sys.stderr)
            except Exception as e:
                print(f"Error initializing sentence transformer: {e}", file=sys.stderr)
                self.has_semantic = False
                self.embedding_model = None
        else:
            self.embedding_model = None
            print("Semantic search not available", file=sys.stderr)
    
    def add_chunks(self, chunk_objects):
        """
        Add chunks to both semantic and keyword indexes.
        
        Args:
            chunk_objects (list): List of chunk objects with text_content and metadata
        """
        if not chunk_objects:
            print("No chunks provided to add", file=sys.stderr)
            return
            
        self.chunk_objects = chunk_objects
        
        # Prepare text content for embedding and indexing
        texts = [chunk['text_content'] for chunk in chunk_objects]
        
        # Build semantic index if dependencies available
        if self.has_semantic and self.has_faiss:
            try:
                # Create embeddings for all chunks
                embeddings = self.embedding_model.encode(texts)
                
                # Normalize the embeddings for cosine similarity
                if np is not None:  # Check if numpy is available
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Initialize and populate vector index
                if faiss is not None:
                    vector_dimension = embeddings.shape[1]
                    self.vector_index = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity
                    self.vector_index.add(embeddings.astype(np.float32))
                    print(f"Added {len(chunk_objects)} chunks to semantic index", file=sys.stderr)
            except Exception as e:
                print(f"Error building semantic index: {e}", file=sys.stderr)
                self.vector_index = None
        
        # Build keyword search index (BM25)
        if self.has_bm25:
            try:
                # Tokenize texts for BM25
                self.tokenized_chunks = [text.lower().split() for text in texts]
                self.bm25 = BM25Okapi(self.tokenized_chunks)
                print(f"Added {len(chunk_objects)} chunks to keyword index", file=sys.stderr)
            except Exception as e:
                print(f"Error building BM25 index: {e}", file=sys.stderr)
                self.bm25 = None
        
        # Store chunk IDs for reference
        self.chunk_ids = list(range(len(chunk_objects)))
    
    def search(self, query, final_top_n=5):
        """
        Search for relevant chunks using hybrid retrieval.
        
        Args:
            query (str): User query
            final_top_n (int): Number of final results to return
            
        Returns:
            list: Most relevant chunks with scores
        """
        if not self.chunk_objects:
            print("No chunks available for search", file=sys.stderr)
            return []
            
        # Step 1: Semantic search (if available)
        semantic_scores = {}
        if self.vector_index and self.has_semantic:
            try:
                # Embed the query
                query_embedding = self.embedding_model.encode(query)
                
                # Normalize for cosine similarity
                if np is not None:
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                # Search vector index
                semantic_scores, semantic_indices = self.vector_index.search(
                    np.array([query_embedding]).astype(np.float32), 
                    min(self.top_k, len(self.chunk_objects))
                )
                
                # Convert to dictionary of {chunk_id: score}
                semantic_scores = {
                    int(idx): float(score) 
                    for score, idx in zip(semantic_scores[0], semantic_indices[0])
                    if idx >= 0 and idx < len(self.chunk_objects)
                }
            except Exception as e:
                print(f"Error in semantic search: {e}", file=sys.stderr)
                semantic_scores = {}
        
        # Step 2: Keyword search (BM25)
        keyword_scores = {}
        if self.bm25 and self.has_bm25:
            try:
                # Tokenize query
                tokenized_query = query.lower().split()
                
                # Get BM25 scores
                bm25_scores = self.bm25.get_scores(tokenized_query)
                
                # Get top-k by score
                if np is not None:
                    top_indices = np.argsort(bm25_scores)[::-1][:self.top_k]
                    
                    # Convert to dictionary of {chunk_id: score}
                    keyword_scores = {
                        idx: float(bm25_scores[idx])
                        for idx in top_indices
                        if idx >= 0 and idx < len(self.chunk_objects)
                    }
                else:
                    # Fallback if numpy not available
                    scores_with_idx = [(score, i) for i, score in enumerate(bm25_scores)]
                    scores_with_idx.sort(reverse=True)
                    
                    keyword_scores = {
                        idx: score 
                        for score, idx in scores_with_idx[:self.top_k]
                        if idx >= 0 and idx < len(self.chunk_objects)
                    }
            except Exception as e:
                print(f"Error in keyword search: {e}", file=sys.stderr)
                keyword_scores = {}
        
        # Step 3: Reciprocal Rank Fusion (RRF) or fallback to available search
        final_scores = {}
        
        # If both methods available, use RRF
        if semantic_scores and keyword_scores:
            all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
            
            for chunk_id in all_ids:
                # Compute reciprocal ranks
                sem_rank = 1.0 / (list(semantic_scores.keys()).index(chunk_id) + self.rrf_k) if chunk_id in semantic_scores else 0
                key_rank = 1.0 / (list(keyword_scores.keys()).index(chunk_id) + self.rrf_k) if chunk_id in keyword_scores else 0
                
                # RRF score is sum of reciprocal ranks
                final_scores[chunk_id] = sem_rank + key_rank
        
        # If only one method available, use its scores
        elif semantic_scores:
            final_scores = semantic_scores
        elif keyword_scores:
            final_scores = keyword_scores
        else:
            # Fallback to basic relevance by term matching if no scoring available
            for i, chunk in enumerate(self.chunk_objects):
                text = chunk["text_content"].lower()
                query_terms = query.lower().split()
                score = sum(1 for term in query_terms if term in text)
                if score > 0:
                    final_scores[i] = score
        
        # Get top-n chunks by final score
        sorted_chunk_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:final_top_n]
        
        # Prepare results with full chunk objects and scores
        results = []
        for chunk_id in sorted_chunk_ids:
            if 0 <= chunk_id < len(self.chunk_objects):
                chunk = self.chunk_objects[chunk_id].copy()
                chunk["score"] = final_scores.get(chunk_id, 0)
                results.append(chunk)
        
        return results
    
    def prepare_context_for_llm(self, selected_chunks, user_query):
        """
        Prepare a formatted context string for the LLM using retrieved chunks.
        
        Args:
            selected_chunks (list): List of chunks from the search method
            user_query (str): The original user query
            
        Returns:
            str: Formatted context string for the LLM
        """
        if not selected_chunks:
            return f"No relevant information found for: {user_query}"
        
        # Sort chunks by chapter/section if available
        try:
            selected_chunks.sort(key=lambda x: (
                x.get("metadata", {}).get("chapter", 0),
                x.get("metadata", {}).get("section", "")
            ))
        except:
            # Skip sorting if metadata format doesn't match
            pass
        
        # Build context string
        context = [f"**Student Question:** {user_query}", "", "**Context from Curriculum:**"]
        
        for i, chunk in enumerate(selected_chunks, 1):
            metadata = chunk.get("metadata", {})
            chapter = metadata.get("chapter", "")
            section = metadata.get("section", "")
            title = metadata.get("title", "")
            
            # Build location string based on available metadata
            location = ""
            if chapter and section:
                location = f"Chapter {chapter}, Section {section}"
            elif chapter:
                location = f"Chapter {chapter}"
            elif title:
                location = title
            else:
                location = f"Chunk {i}"
            
            context.append(f"--- {location}: {chunk['text_content'][:300]}...")
        
        context.append("")
        context.append("**Your Answer:**")
        
        return "\n".join(context)
