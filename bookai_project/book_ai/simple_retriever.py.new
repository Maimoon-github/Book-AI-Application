# filepath: d:\maimoon\Vs Code\Book-AI-Application\Book-AI-Application\bookai_project\book_ai\simple_retriever.py
import warnings
import sys

# Track dependency status
DEPENDENCIES_STATUS = {
    "sentence_transformers": False,
    "numpy": False,
    "rank_bm25": False
}

# Try to import dependencies with robust fallbacks
try:
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_STATUS["sentence_transformers"] = True
except ImportError:
    warnings.warn("sentence_transformers not available, using dummy implementation")
    # Create a dummy SentenceTransformer class
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, *args, **kwargs):
            return [[0.0] * 384]  # Default embedding dimension

try:
    import numpy as np
    DEPENDENCIES_STATUS["numpy"] = True
except ImportError:
    warnings.warn("numpy not available, retrieval functionality will be limited")
    # Create minimal dummy implementation with linalg submodule
    class np:
        class linalg:
            @staticmethod
            def norm(x, axis=None, keepdims=False):
                return 1.0
        
        @staticmethod
        def zeros(*args, **kwargs):
            return [[0.0]]
        
        @staticmethod
        def dot(a, b):
            return 0.0
        
        @staticmethod
        def divide(a, b, **kwargs):
            return a
        
        @staticmethod
        def zeros_like(x):
            return x
        
        @staticmethod
        def argsort(x):
            if isinstance(x, list):
                return list(range(min(len(x), 5)))
            return [0]

try:
    from rank_bm25 import BM25Okapi
    DEPENDENCIES_STATUS["rank_bm25"] = True
except ImportError:
    warnings.warn("rank_bm25 not available, keyword search will be disabled")
    # Create dummy BM25Okapi class
    class BM25Okapi:
        def __init__(self, *args, **kwargs):
            pass
        def get_scores(self, *args, **kwargs):
            return [0.0]  # Return dummy scores

# Log dependency status
print("SimpleHybridRetriever Dependencies:", file=sys.stderr)
for dep, status in DEPENDENCIES_STATUS.items():
    status_str = "✓ Available" if status else "✗ Missing"
    print(f"  - {dep}: {status_str}", file=sys.stderr)

class SimpleHybridRetriever:
    """
    A simplified hybrid retrieval system that works without FAISS.
    Includes fallbacks for when dependencies aren't available.
    """
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', top_k=20, rrf_k=60):
        self.model_name = model_name
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.chunk_objects = []
        self.embeddings = []
        self.tokenized_chunks = []
        self.bm25 = None
        
        # Initialize embedding model if available
        if DEPENDENCIES_STATUS["sentence_transformers"]:
            try:
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                warnings.warn(f"Error initializing SentenceTransformer: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            
    def add_chunks(self, chunk_objects):
        """Add chunks to the retrieval system and build indices."""
        self.chunk_objects = chunk_objects
        
        if not chunk_objects:
            return
            
        texts = [chunk['text_content'] for chunk in chunk_objects]
        
        # Create embeddings for all chunks if the model is available
        if self.embedding_model is not None:
            try:
                self.embeddings = self.embedding_model.encode(texts)
                
                if DEPENDENCIES_STATUS["numpy"]:
                    # Normalize embeddings for cosine similarity using numpy
                    norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                    mask = norms > 0
                    self.embeddings = np.divide(self.embeddings, norms, out=np.zeros_like(self.embeddings), where=mask)
            except Exception as e:
                warnings.warn(f"Error creating embeddings: {e}")
                self.embeddings = []
                
        # Initialize and populate keyword search index (BM25) if available
        if DEPENDENCIES_STATUS["rank_bm25"]:
            try:
                self.tokenized_chunks = [text.lower().split() for text in texts]
                self.bm25 = BM25Okapi(self.tokenized_chunks)
            except Exception as e:
                warnings.warn(f"Error initializing BM25: {e}")
                self.tokenized_chunks = []
                self.bm25 = None
                
    def search(self, query, final_top_n=5):
        """Perform hybrid search using both semantic and keyword methods with fallbacks."""
        if not self.chunk_objects:
            return []
        
        semantic_indices = []
        semantic_ranks = {}
        keyword_indices = []
        keyword_ranks = {}
        
        # Perform semantic search if possible
        if self.embedding_model is not None and len(self.embeddings) > 0 and DEPENDENCIES_STATUS["numpy"]:
            try:
                # Embed the user query
                query_embedding = self.embedding_model.encode([query])[0]
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
                
                # Compute cosine similarity with numpy
                similarities = np.dot(self.embeddings, query_embedding)
                semantic_indices = np.argsort(similarities)[::-1][:self.top_k]
                semantic_ranks = {idx: rank + 1 for rank, idx in enumerate(semantic_indices)}
            except Exception as e:
                warnings.warn(f"Error in semantic search: {e}")
                semantic_indices = []
                semantic_ranks = {}
        
        # Perform keyword search if possible
        if self.bm25 is not None and DEPENDENCIES_STATUS["rank_bm25"] and DEPENDENCIES_STATUS["numpy"]:
            try:
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                keyword_indices = np.argsort(bm25_scores)[::-1][:self.top_k]
                keyword_ranks = {idx: rank + 1 for rank, idx in enumerate(keyword_indices)}
            except Exception as e:
                warnings.warn(f"Error in keyword search: {e}")
                keyword_indices = []
                keyword_ranks = {}
        
        # Create initial pool of chunk IDs
        combined_indices = set(semantic_indices) | set(keyword_indices)
        
        # If no indices were found through either method, return a random sample
        if not combined_indices:
            import random
            sample_size = min(final_top_n, len(self.chunk_objects))
            return random.sample(self.chunk_objects, sample_size) if sample_size > 0 else []
        
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
