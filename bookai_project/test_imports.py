# Test if imports are working correctly
import sys
print(f"Python version: {sys.version}")

# Test all required dependencies
dependencies = {
    "numpy": False,
    "PyPDF2": False,
    "faiss": False,
    "torch": False,
    "nltk": False,
    "sentence_transformers": False,
    "rank_bm25": False,
}

# Try importing numpy
try:
    import numpy as np
    dependencies["numpy"] = True
    print("numpy version:", np.__version__)
except Exception as e:
    print(f"numpy error: {str(e)}")

# Try importing PyPDF2
try:
    import PyPDF2
    dependencies["PyPDF2"] = True
    print("PyPDF2 version:", PyPDF2.__version__)
except Exception as e:
    print(f"PyPDF2 error: {str(e)}")

# Try importing faiss
try:
    import faiss
    dependencies["faiss"] = True
    print("faiss available:", "Yes")
except Exception as e:
    print(f"faiss error: {str(e)}")

# Try importing torch
try:
    import torch
    dependencies["torch"] = True
    print("torch version:", torch.__version__)
except Exception as e:
    print(f"torch error: {str(e)}")

# Try importing nltk
try:
    import nltk
    dependencies["nltk"] = True
    print("nltk version:", nltk.__version__)
except Exception as e:
    print(f"nltk error: {str(e)}")

# Try importing sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    dependencies["sentence_transformers"] = True
    print("sentence_transformers available:", "Yes")
except Exception as e:
    print(f"sentence_transformers error: {str(e)}")

# Try importing rank_bm25
try:
    from rank_bm25 import BM25Okapi
    dependencies["rank_bm25"] = True
    print("rank_bm25 available:", "Yes")
except Exception as e:
    print(f"rank_bm25 error: {str(e)}")

print("\nSummary of dependencies:")
for dep, status in dependencies.items():
    print(f"  - {dep}: {'✓ Available' if status else '✗ Missing'}")

# Test importing from local modules
print("\nTesting local module imports:")
try:
    from book_ai.rag_utils import HierarchicalChunker, MetadataEnricher, HybridRetriever
    print("Successfully imported HierarchicalChunker, MetadataEnricher, and HybridRetriever from book_ai.rag_utils")
except Exception as e:
    print(f"Error importing from book_ai.rag_utils: {str(e)}")
