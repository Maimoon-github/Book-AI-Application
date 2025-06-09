import numpy as np
import warnings
import re
import math
import uuid
from collections import Counter
import json
import sys

# Import core dependencies with error handling
DEPENDENCIES_STATUS = {
    "faiss": False,
    "nltk": False, 
    "sentence_transformers": False,
    "rank_bm25": False,
    "torch": False,
    "PyPDF2": False
}

# Standard libraries first
try:
    from rank_bm25 import BM25Okapi
    DEPENDENCIES_STATUS["rank_bm25"] = True
except ImportError:
    warnings.warn("rank_bm25 not found. Keyword-based search will be disabled.")
    # Create a dummy BM25Okapi class for graceful degradation
    class BM25Okapi:
        def __init__(self, *args, **kwargs):
            warnings.warn("Using dummy BM25Okapi implementation.")
        def get_scores(self, *args, **kwargs):
            return np.zeros(0)

try:
    from sentence_transformers import SentenceTransformer, util
    DEPENDENCIES_STATUS["sentence_transformers"] = True
except ImportError:
    warnings.warn("sentence_transformers not found. Semantic search will be disabled.")
    # Create dummy implementations
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            warnings.warn("Using dummy SentenceTransformer implementation.")
        def encode(self, *args, **kwargs):
            return np.zeros((1, 384))  # Default embedding size
    
    class util:
        @staticmethod
        def cos_sim(*args, **kwargs):
            return 0.0

try:
    import torch
    DEPENDENCIES_STATUS["torch"] = True
except ImportError:
    warnings.warn("PyTorch not found. Some operations may be limited.")

# Try to import FAISS with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
    DEPENDENCIES_STATUS["faiss"] = True
except ImportError:
    warnings.warn("FAISS not found. Using slower numpy-based similarity search as fallback.")
    FAISS_AVAILABLE = False

# Try to import NLTK components
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize
    DEPENDENCIES_STATUS["nltk"] = True
    
    # Download necessary NLTK data (only needed once)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
except ImportError:
    warnings.warn("NLTK not found. Text tokenization and stopword removal will be limited.")
    # Create dummy functions/classes
    def sent_tokenize(text):
        return [text]  # Return the whole text as one sentence
    stopwords = type('', (), {'words': lambda *args: []})()

# Try to import PDF handling library
try:
    import PyPDF2
    DEPENDENCIES_STATUS["PyPDF2"] = True
except ImportError:
    warnings.warn("PyPDF2 not found. PDF processing will be disabled.")
    # Create a dummy class with the necessary interface
    class PdfReader:
        def __init__(self, *args, **kwargs):
            self.pages = []
    
    class PyPDF2:
        PdfReader = PdfReader

# Print dependency status
print("RAG Dependencies Status:", file=sys.stderr)
for dep, status in DEPENDENCIES_STATUS.items():
    status_str = "✓ Available" if status else "✗ Missing"
    print(f"  - {dep}: {status_str}", file=sys.stderr)

class HierarchicalChunker:
    """
    Splits document content into semantic chunks based on chapters and context.
    """
    def __init__(self, semantic_threshold=0.6, overlap_sentences=1, max_chunk_sentences=15,
                model_name='paraphrase-MiniLM-L6-v2'):
        self.semantic_threshold = semantic_threshold
        self.overlap_sentences = overlap_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.model = SentenceTransformer(model_name)
    
    def extract_text_from_pdf(self, file):
        """Extract text content from a PDF file."""
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def extract_chapters(self, content):
        """Extracts individual chapters from the content."""
        chapter_pattern = r'(?:Chapter|CHAPTER)\s+(\d+)[\s:-]+\s*([^\n]+)'
        chapter_matches = list(re.finditer(chapter_pattern, content))
        chapters_data = []

        for i, match in enumerate(chapter_matches):
            chapter_number = int(match.group(1))
            chapter_title = match.group(2).strip()
            start_pos = match.start()
            end_pos = chapter_matches[i+1].start() if i < len(chapter_matches) - 1 else len(content)
            chapter_text = content[start_pos:end_pos].strip()

            chapters_data.append({
                "number": chapter_number,
                "title": chapter_title,
                "text": chapter_text
            })

        if not chapters_data:
            chapters_data.append({
                "number": 1,
                "title": "Content",
                "text": content
            })

        return chapters_data
    
    def semantic_chunk_chapter(self, chapter_text, chapter_number, chapter_title):
        """Divides a chapter into semantically coherent chunks."""
        sentences = sent_tokenize(chapter_text)
        if len(sentences) == 0:
            return []

        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)

        chunks = []
        current_chunk_sentences = []
        current_sentence_count = 0

        for i in range(len(sentences)):
            current_chunk_sentences.append(sentences[i])
            current_sentence_count += 1

            # Create a new chunk if max size reached or semantic boundary detected
            create_new_chunk = current_sentence_count >= self.max_chunk_sentences

            if i < len(sentences) - 1 and not create_new_chunk and current_sentence_count >= 3:
                # Check semantic similarity with next sentence
                similarity = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i+1]).item()
                if similarity < self.semantic_threshold:
                    create_new_chunk = True

            if create_new_chunk or i == len(sentences) - 1:
                # Add the current chunk
                chunks.append({
                    "text_content": " ".join(current_chunk_sentences),
                    "metadata": {
                        "chapter_number": chapter_number,
                        "chapter_title": chapter_title,
                        "chunk_index": len(chunks)
                    }
                })

                # Start a new chunk with overlap
                if i < len(sentences) - 1:
                    overlap_start = max(0, len(current_chunk_sentences) - self.overlap_sentences)
                    current_chunk_sentences = current_chunk_sentences[overlap_start:]
                    current_sentence_count = len(current_chunk_sentences)

        return chunks
    
    def process_document(self, file_obj):
        """Process a document and return chunked content."""
        content = self.extract_text_from_pdf(file_obj)
        if not content:
            raise ValueError("Could not extract text from the document")
        
        # Extract chapters and chunk them
        chapters = self.extract_chapters(content)
        all_chunks = []
        
        for chapter in chapters:
            chapter_chunks = self.semantic_chunk_chapter(
                chapter["text"],
                chapter["number"],
                chapter["title"]
            )
            all_chunks.extend(chapter_chunks)
        
        return all_chunks

class MetadataEnricher:
    """
    Enriches chunks with detailed metadata for better retrieval.
    """
    def __init__(self):
        # Make sure NLTK resources are available
        try:
            stopwords.words('english')
            sent_tokenize("Test sentence.")
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
    
    def enrich_chunks(self, chunks):
        """Add detailed metadata to chunks for better retrieval."""
        enriched_chunks = chunks.copy()
        stop_words = set(stopwords.words('english'))
        section_pattern = r'(?:^|\n)(\d+\.\d+(?:\.\d+)*)\s+([^\n]+)'
        
        # Dictionary for TF-IDF calculation
        all_words = Counter()
        chunk_word_counts = []
        
        # First pass: gather word counts
        for chunk in chunks:
            text = chunk["text_content"]
            words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) 
                    if w.lower() not in stop_words]
            chunk_words = Counter(words)
            all_words.update(chunk_words)
            chunk_word_counts.append(chunk_words)
        
        # Calculate IDF
        num_chunks = len(chunks)
        word_idf = {}
        for word, count in all_words.items():
            chunks_with_word = sum(1 for chunk_words in chunk_word_counts if word in chunk_words)
            word_idf[word] = math.log(num_chunks / (1 + chunks_with_word))
        
        # Second pass: augment metadata
        for i, chunk in enumerate(enriched_chunks):
            text = chunk["text_content"]
            chunk_id = str(uuid.uuid4())
            
            # Initialize with existing metadata
            enhanced_metadata = chunk["metadata"].copy()
            enhanced_metadata["chunk_id"] = chunk_id
            
            # Extract section information
            section_match = re.search(section_pattern, text)
            if section_match:
                enhanced_metadata["section_number"] = section_match.group(1)
                enhanced_metadata["section_title"] = section_match.group(2).strip()
            
            # Extract keywords using TF-IDF
            chunk_words = chunk_word_counts[i]
            word_scores = {word: count * word_idf[word] for word, count in chunk_words.items()}
            keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            enhanced_metadata["keywords"] = [word for word, _ in keywords]
            
            # Estimate difficulty level
            avg_word_length = sum(len(word) for word in text.split()) / max(1, len(text.split()))
            sentences = sent_tokenize(text)
            avg_sentence_length = sum(len(sent.split()) for sent in sentences) / max(1, len(sentences))
            
            if avg_word_length < 5 and avg_sentence_length < 10:
                difficulty = "beginner"
            elif avg_word_length < 6 and avg_sentence_length < 15:
                difficulty = "intermediate"
            else:
                difficulty = "advanced"
                
            enhanced_metadata["difficulty_level"] = difficulty
            
            # Add learning objective
            chapter_title = enhanced_metadata.get("chapter_title", "")
            enhanced_metadata["learning_objective"] = f"Understand concepts related to {chapter_title}"
            
            # Update chunk with enhanced metadata
            chunk["metadata"] = enhanced_metadata
        
        return enriched_chunks

class HybridRetriever:
    """
    Hybrid retrieval system combining semantic search and keyword-based search.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", top_k=10, rrf_k=60):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.chunk_objects = []
        self.vector_index = None
        self.chunk_ids = []
        self.bm25 = None
        self.tokenized_chunks = []

    def add_chunks(self, chunk_objects):
        """Add chunks to both semantic and keyword indexes."""
        self.chunk_objects = chunk_objects
        
        # Prepare text content for embedding and indexing
        texts = [chunk['text_content'] for chunk in chunk_objects]
        
        # Create embeddings for all chunks
        embeddings = self.embedding_model.encode(texts)
        
        # Normalize the embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Initialize and populate FAISS vector index
        vector_dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(vector_dimension)  # Inner Product for cosine similarity
        self.vector_index.add(embeddings.astype(np.float32))
        
        # Store chunk IDs for reference
        self.chunk_ids = [i for i in range(len(chunk_objects))]
        
        # Initialize and populate keyword search index (BM25)
        self.tokenized_chunks = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query, final_top_n=5):
        """Perform hybrid search using both semantic and keyword methods."""
        if not self.chunk_objects:
            return []
            
        # Embed the user query
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Perform semantic search
        semantic_scores, semantic_indices = self.vector_index.search(
            np.array([query_embedding]).astype(np.float32), 
            self.top_k
        )
        semantic_indices = semantic_indices[0]
        semantic_ranks = {idx: rank + 1 for rank, idx in enumerate(semantic_indices)}
        
        # Perform keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        keyword_indices = np.argsort(bm25_scores)[::-1][:self.top_k]
        keyword_ranks = {idx: rank + 1 for rank, idx in enumerate(keyword_indices)}
        
        # Create initial pool of chunk IDs
        combined_indices = set(semantic_indices) | set(keyword_indices)
        
        # Perform Reciprocal Rank Fusion
        rrf_scores = {}
        for idx in combined_indices:
            # Get ranks (use top_k+1 if not found in one of the result sets)
            semantic_rank = semantic_ranks.get(idx, self.top_k + 1)
            keyword_rank = keyword_ranks.get(idx, self.top_k + 1)
            
            # Calculate RRF score
            rrf_score = 1/(self.rrf_k + semantic_rank) + 1/(self.rrf_k + keyword_rank)
            rrf_scores[idx] = rrf_score
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        
        # Select top N results
        final_indices = sorted_indices[:final_top_n]
        
        # Return the selected chunk objects
        return [self.chunk_objects[idx] for idx in final_indices]
    
    def prepare_context_for_llm(self, selected_chunks, user_query):
        """Prepares context from selected chunks for the LLM."""
        context_parts = []
        
        for i, chunk in enumerate(selected_chunks, 1):
            # Extract source information from metadata
            source_info = ""
            meta = chunk['metadata']
            if 'chapter_number' in meta and 'section_number' in meta:
                source_info = f"(Chapter {meta['chapter_number']}, Section {meta['section_number']})"
            elif 'chapter_number' in meta:
                source_info = f"(Chapter {meta['chapter_number']})"
            
            context_parts.append(f"--- Chunk {i} {source_info}: {chunk['text_content']}")
        
        context_text = "\n".join(context_parts)
        
        prompt = f"""
You are an AI teacher. Answer the student's question concisely and accurately, *only* using the provided context. If the answer is not found in the context, state that you don't know.

**Student Question:** {user_query}

**Context from Curriculum:**
{context_text}

**Your Answer:**
"""
        return prompt
