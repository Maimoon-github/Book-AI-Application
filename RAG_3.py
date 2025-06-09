import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch

class HybridRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", top_k=10, rrf_k=60):
        """
        Initialize the hybrid retriever with semantic and keyword search capabilities.
        
        Args:
            model_name (str): Name of the sentence embedding model to use
            top_k (int): Number of top results to retrieve from each search method
            rrf_k (int): Constant for Reciprocal Rank Fusion calculation
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.chunk_objects = []
        
        # Vector database for semantic search
        self.vector_index = None
        self.chunk_ids = []
        
        # Keyword search index
        self.bm25 = None
        self.tokenized_chunks = []

    def add_chunks(self, chunk_objects):
        """
        Add chunks to both semantic and keyword indexes.
        
        Args:
            chunk_objects (list): List of chunk objects with text_content and metadata
        """
        self.chunk_objects = chunk_objects
        
        # Prepare text content for embedding and indexing
        texts = [chunk['text_content'] for chunk in chunk_objects]
        
        # Step 1.1: Create embeddings for all chunks
        embeddings = self.embedding_model.encode(texts)
        
        # Normalize the embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Step 1.2: Initialize and populate FAISS vector index
        vector_dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(vector_dimension)  # Inner Product for cosine similarity
        self.vector_index.add(embeddings.astype(np.float32))
        
        # Store chunk IDs for reference
        self.chunk_ids = [i for i in range(len(chunk_objects))]
        
        # Step 1.3: Initialize and populate keyword search index (BM25)
        self.tokenized_chunks = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        print(f"Added {len(chunk_objects)} chunks to hybrid retriever")

    def search(self, query, final_top_n=5):
        """
        Perform hybrid search using both semantic and keyword methods.
        
        Args:
            query (str): User's query text
            final_top_n (int): Number of final results to return after fusion
            
        Returns:
            list: Top N most relevant chunk objects
        """
        if not self.chunk_objects:
            return []
            
        # Step 2.1: Embed the user query
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Step 2.2: Perform semantic search
        semantic_scores, semantic_indices = self.vector_index.search(
            np.array([query_embedding]).astype(np.float32), 
            self.top_k
        )
        semantic_indices = semantic_indices[0]
        semantic_ranks = {idx: rank + 1 for rank, idx in enumerate(semantic_indices)}
        
        # Step 2.3: Perform keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        keyword_indices = np.argsort(bm25_scores)[::-1][:self.top_k]
        keyword_ranks = {idx: rank + 1 for rank, idx in enumerate(keyword_indices)}
        
        # Step 2.4: Create initial pool of chunk IDs (combining both result sets)
        combined_indices = set(semantic_indices) | set(keyword_indices)
        
        # Step 3: Perform Reciprocal Rank Fusion
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
        
        # Step 3.3: Select top N results
        final_indices = sorted_indices[:final_top_n]
        
        # Return the selected chunk objects
        return [self.chunk_objects[idx] for idx in final_indices]
    
    def prepare_context_for_llm(self, selected_chunks, user_query):
        """
        Prepares context from selected chunks for the LLM.
        
        Args:
            selected_chunks (list): List of selected chunk objects
            user_query (str): Original user query
            
        Returns:
            str: Formatted prompt with context for LLM
        """
        context_parts = []
        
        for i, chunk in enumerate(selected_chunks, 1):
            # Extract source information from metadata
            source_info = ""
            if 'metadata' in chunk:
                meta = chunk['metadata']
                if 'chapter' in meta and 'section' in meta:
                    source_info = f"(Chapter {meta['chapter']}, Section {meta['section']})"
                elif 'source' in meta:
                    source_info = f"({meta['source']})"
            
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

# Example usage
if __name__ == "__main__":
    # Sample chunks for demonstration
    sample_chunks = [
        {
            "text_content": "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
            "metadata": {"chapter": 1, "section": "1.2", "page": 5}
        },
        {
            "text_content": "Plants make their food using chlorophyll, which gives them their green color.",
            "metadata": {"chapter": 1, "section": "1.2", "page": 6}
        },
        {
            "text_content": "The mitochondria is the powerhouse of the cell.",
            "metadata": {"chapter": 2, "section": "2.1", "page": 12}
        },
        {
            "text_content": "During photosynthesis, plants take in carbon dioxide and release oxygen.",
            "metadata": {"chapter": 1, "section": "1.3", "page": 7}
        }
    ]
    
    # Initialize hybrid retriever
    retriever = HybridRetriever()
    
    # Add chunks to both indexes
    retriever.add_chunks(sample_chunks)
    
    # Search with a user query
    user_query = "How do plants make food?"
    relevant_chunks = retriever.search(user_query, final_top_n=3)
    
    # Prepare context for LLM
    llm_prompt = retriever.prepare_context_for_llm(relevant_chunks, user_query)
    
    # Display results
    print("\nTop relevant chunks:")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"{i}. {chunk['text_content']}")
    
    print("\nPrompt for LLM:")
    print(llm_prompt)