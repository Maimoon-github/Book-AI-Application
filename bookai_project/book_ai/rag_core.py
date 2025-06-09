# book_ai/rag_core.py

from django.conf import settings
import os
import json
import sys
import warnings
import importlib.util
from .models import Document, Chunk

# Track dependency status
DEPENDENCIES_STATUS = {
    "faiss": False,
    "nltk": False, 
    "sentence_transformers": False,
    "rank_bm25": False,
    "torch": False,
    "PyPDF2": False,
    "numpy": False,
    "transformers": False
}

# Try importing different retriever implementations in order of decreasing complexity
# Start with the most feature-rich and fall back to simpler ones if dependencies are missing
retriever_type = "basic"  # Default to the simplest version

# Check if package is installed
def is_package_available(package_name):
    """Check if a Python package is available without importing it"""
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ImportError, AttributeError, ValueError):
        return False

# Detect available packages
for package in DEPENDENCIES_STATUS:
    DEPENDENCIES_STATUS[package] = is_package_available(package)
    status_str = "✓ Available" if DEPENDENCIES_STATUS[package] else "✗ Missing"
    print(f"Package {package}: {status_str}", file=sys.stderr)

try:
    # Try the full FAISS-based hybrid retriever first
    if DEPENDENCIES_STATUS["faiss"] and DEPENDENCIES_STATUS["sentence_transformers"]:
        from .rag_utils import HybridRetriever
        RetrieverClass = HybridRetriever
        retriever_type = "faiss"
        print("Using FAISS-based HybridRetriever", file=sys.stderr)
    else:
        raise ImportError("FAISS or sentence_transformers missing")
except ImportError as e:
    print(f"FAISS retriever not available: {e}", file=sys.stderr)
    try:
        # Try the numpy-based hybrid retriever with BM25
        if DEPENDENCIES_STATUS["numpy"] and DEPENDENCIES_STATUS["rank_bm25"] and DEPENDENCIES_STATUS["sentence_transformers"]:
            from .simple_retriever import SimpleHybridRetriever
            RetrieverClass = SimpleHybridRetriever
            retriever_type = "simple"
            print("Using SimpleHybridRetriever with BM25", file=sys.stderr)
        else:
            raise ImportError("numpy, rank_bm25, or sentence_transformers missing")
    except ImportError as e:
        print(f"SimpleHybridRetriever not available: {e}", file=sys.stderr)
        # Fall back to the most basic retriever
        from .basic_retriever import BasicRetriever
        RetrieverClass = BasicRetriever
        print("Using minimal BasicRetriever", file=sys.stderr)

# Initialize a global retriever
try:
    retriever = RetrieverClass()
    print(f"Successfully initialized {RetrieverClass.__name__}", file=sys.stderr)
except Exception as e:
    print(f"Error initializing retriever: {e}", file=sys.stderr)
    # Create an emergency fallback retriever in case all else fails
    from .basic_retriever import BasicRetriever
    retriever = BasicRetriever()
    print("Using emergency fallback retriever", file=sys.stderr)

initialized = False

# Import transformers conditionally to avoid startup errors if not installed
TRANSFORMERS_AVAILABLE = False
try:
    if DEPENDENCIES_STATUS["transformers"]:
        from transformers import pipeline
        TRANSFORMERS_AVAILABLE = True
    else:
        warnings.warn("Transformers library not available. Text generation will be limited.")
except ImportError:
    warnings.warn("Transformers library not available. Text generation will be limited.")

def initialize_retriever():
    """Initialize the retriever with chunks from the database."""
    global initialized
    try:
        # Convert DB chunks to the format expected by HybridRetriever
        all_chunks = []
        
        # Get all chunks from the database
        db_chunks = Chunk.objects.all().select_related('document')
        
        print(f"Found {len(db_chunks)} chunks in database", file=sys.stderr)
        
        for db_chunk in db_chunks:
            try:
                # Ensure metadata is a valid dict
                metadata = {}
                if db_chunk.metadata:
                    try:
                        if isinstance(db_chunk.metadata, dict):
                            metadata = db_chunk.metadata
                        elif isinstance(db_chunk.metadata, str):
                            metadata = json.loads(db_chunk.metadata)
                    except Exception as e:
                        print(f"Error parsing metadata: {e}", file=sys.stderr)
                
                chunk_obj = {
                    'text_content': db_chunk.text_content or "",
                    'metadata': metadata,
                    'db_id': db_chunk.pk  # Keep track of DB ID
                }
                all_chunks.append(chunk_obj)
            except Exception as e:
                print(f"Error processing chunk {db_chunk.pk}: {e}", file=sys.stderr)
        
        # Add chunks to the retriever
        if all_chunks:
            try:
                retriever.add_chunks(all_chunks)
                initialized = True
                print(f"Successfully added {len(all_chunks)} chunks to retriever", file=sys.stderr)
                return True
            except Exception as e:
                print(f"Error adding chunks to retriever: {e}", file=sys.stderr)
                return False
        else:
            print("No chunks found to initialize retriever", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Error initializing retriever: {e}", file=sys.stderr)
        return False

def get_answer(query):
    """Gets an answer using the retriever and a simple text generation model."""
    global initialized
    if not initialized:
        try:
            initialized = initialize_retriever()
        except Exception as e:
            print(f"Error initializing retriever: {e}", file=sys.stderr)
            return f"Error initializing the retrieval system: {e}"
    
    if not initialized or not query:
        return "Sorry, no documents have been uploaded to answer questions from."
    
    # Get relevant chunks using the retriever safely
    try:
        relevant_chunks = retriever.search(query, final_top_n=3)
    except Exception as e:
        print(f"Error during retrieval: {e}", file=sys.stderr)
        return f"Error during retrieval: {e}"
    
    if not relevant_chunks:
        return "No relevant information found to answer this question."
    
    # Try to use transformers if available
    answer = None
    if TRANSFORMERS_AVAILABLE and retriever_type != "basic":
        try:
            # This would use transformers to generate an answer if available
            # For now, just return the same format as below
            pass
        except Exception as e:
            print(f"Error using text generation model: {e}", file=sys.stderr)
    
    # Fall back to simple text concatenation if transformers not available or fails
    if answer is None:
        try:
            answer_parts = []
            for chunk in relevant_chunks:
                # Extract metadata info
                meta = chunk.get('metadata', {})
                source_info = ""
                if 'chapter_number' in meta and 'section_number' in meta:
                    source_info = f" (From Chapter {meta['chapter_number']}, Section {meta['section_number']})"
                elif 'chapter_title' in meta:
                    source_info = f" (From chapter on '{meta['chapter_title']}')"
                
                answer_parts.append(f"{chunk['text_content']}{source_info}")
            
            answer = "\n\n".join(answer_parts)
        except Exception as e:
            print(f"Error formatting answer: {e}", file=sys.stderr)
            # Last resort fallback - return raw text if metadata extraction fails
            answer = "\n\n".join([c.get('text_content', 'No content available') for c in relevant_chunks])
    
    return "Based on the document content:\n\n" + answer