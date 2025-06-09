import warnings

# Try to import dependencies with robust fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence_transformers not available, using dummy implementation")
    # Create a dummy SentenceTransformer class
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, *args, **kwargs):
            return [[0.0] * 384]  # Default embedding dimension

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available, retrieval functionality will be limited")
    # Create minimal dummy implementation
    class np:
        @staticmethod
        def zeros(*args, **kwargs):
            return [[0.0]]
        
        @staticmethod
        def dot(a, b):
            return 0.0
        
        @staticmethod
        def linalg_norm(x, *args, **kwargs):
            return 1.0
        
        @staticmethod
        def divide(a, b, **kwargs):
            return a
        
        @staticmethod
        def zeros_like(x):
            return x
        
        @staticmethod
        def argsort(x):
            return [0]

class BasicRetriever:
    """
    A basic retrieval system that only uses cosine similarity with numpy.
    Designed to work with minimal dependencies when FAISS and BM25 aren't available.
    """
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', top_k=20):
        self.model_name = model_name
        self.top_k = top_k
        self.chunk_objects = []
        self.embeddings = []
        
        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                warnings.warn(f"Error initializing SentenceTransformer: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
    
    def add_chunks(self, chunk_objects):
        """Add chunks to the retrieval system."""
        self.chunk_objects = chunk_objects
        if not chunk_objects:
            return
            
        texts = [chunk['text_content'] for chunk in chunk_objects]
        
        # Create embeddings for all chunks
        self.embeddings = self.embedding_model.encode(texts)
          # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        mask = norms > 0
        self.embeddings = np.divide(self.embeddings, norms, out=np.zeros_like(self.embeddings), where=mask)

    def search(self, query, final_top_n=5):
        """Perform simple cosine similarity search."""
        if not self.chunk_objects or len(self.embeddings) == 0:
            return []
            
        # Embed the user query
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarity with numpy
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get the indices of top results
        top_indices = np.argsort(similarities)[::-1][:final_top_n]
        
        # Return the selected chunk objects
        return [self.chunk_objects[idx] for idx in top_indices]
