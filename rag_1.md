This document outlines a single, step-by-step functional plan to implement a Retrieval-Augmented Generation (RAG) pipeline for your AI teacher chatbot, integrating Hierarchical Chunking, Metadata-Augmentation, and Hybrid Search. This comprehensive approach ensures highly relevant and context-rich information retrieval from your chapter-based educational materials.

### **Overall Goal:**

To transform raw educational content into an efficiently searchable knowledge base and, upon a user query, retrieve the most precise and relevant information to augment a Large Language Model (LLM) for generating accurate AI teacher responses.

### **Inputs:**

* Raw educational text (e.g., a book or course material, potentially as a single large string or a collection of chapter files).  
* User queries (natural language questions from students).

### **Outputs:**

* A knowledge base (vector database \+ keyword index) populated with well-structured, metadata-rich chunks.  
* For each user query, a highly ranked list of the most relevant chunks, formatted as context for your generative LLM.

### **Prerequisites (Tools/Libraries):**

* **Python 3.x**  
* **Text Processing:** NLTK (for sentence tokenization), SpaCy (optional, for more advanced NLP like entity recognition if desired).  
* **Embedding Models:** sentence-transformers library (e.g., all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2, or a larger model like BAAI/bge-small-en-v1.5).  
* **Vector Database:** Choose one (e.g., Faiss for local, Pinecone, ChromaDB, Weaviate, Milvus for cloud/production).  
* **Keyword Search:** rank\_bm25 library for a simple BM25 implementation.  
* **LLM Access:** An API key/access to a generative LLM (e.g., Gemini-2.0-Flash) for keyword extraction and final answer generation.  
* **Data Structures:** Standard Python lists and dictionaries.

### **Step-by-Step Functional Execution Plan:**

#### **Phase 1: Content Preparation (Hierarchical Chunking & Metadata Augmentation)**

This phase focuses on structuring and enriching your raw educational content.

**Step 1.1: Load Raw Content & Identify Chapters**

* **Action:** Read your educational content. If it's a single large file, programmatically identify and separate distinct chapters. If each chapter is already a separate file, load them individually.  
* **Method:**  
  * Define a clear pattern for chapter boundaries (e.g., "Chapter N:", "\# Chapter Title").  
  * Iterate through the raw text or files, extracting the full text content for each chapter.  
  * For each extracted chapter, identify and store its chapter\_number and chapter\_title.  
* **Output:** A list of ChapterData objects/dictionaries: \[ { "number": 1, "title": "Introduction", "text": "..." }, ... \]

**Step 1.2: Semantic Sub-Chunking within Chapters**

* **Action:** For each chapter's full text, break it down into smaller, semantically coherent chunks.  
* **Method:**  
  1. **Sentence Segmentation:** Use nltk.sent\_tokenize() (ensure you've downloaded punkt tokenizer: nltk.download('punkt')) to split the chapter text into individual sentences.  
  2. **Sentence Embedding:** Load your chosen sentence-transformers model. Generate embeddings for *each sentence*.  
  3. **Semantic Boundary Detection:** Iterate through the sentence embeddings. Calculate the cosine similarity between adjacent sentence embeddings. When the similarity drops below a predefined semantic\_threshold (e.g., 0.6 to 0.7), mark it as a potential chunk boundary. This indicates a shift in topic.  
  4. **Chunk Formation:** Group sentences between these detected boundaries into a single semantic chunk.  
  5. **Overlap (Crucial):** Implement a small overlap (e.g., 1-3 sentences) between consecutive chunks. This ensures that context isn't lost if a key idea spans a boundary.  
* **Output:** A list of initial ChunkData objects/dictionaries: \[ { "content": "sentence 1\. sentence 2.", "metadata": { "chapter\_number": 1, "chapter\_title": "Introduction" } }, ... \]

**Step 1.3: Augment Chunks with Detailed Metadata**

* **Action:** Enhance each semantic chunk with rich, searchable metadata. This is where the "AI teacher" aspect comes in, enabling more precise retrieval.  
* **Method:**  
  1. **Unique Chunk ID:** Assign a unique\_id (e.g., UUID) to each chunk.  
  2. **Section/Sub-heading (if applicable):** If your chapters have consistent section headings (e.g., "1.1 Introduction"), parse these during chunking and add section\_title and section\_number to the metadata of relevant chunks.  
  3. **Keyword/Topic Extraction (LLM-based):**  
     * For each chunk's content, make an API call to a generative LLM (like Gemini-2.0-Flash).  
     * **Prompt:** "Given the following educational text, identify 3-7 key concepts or keywords that are central to its meaning. List them as a comma-separated string, e.g., 'concept1, concept2, keyword3'."  
       {  
           "contents": \[  
               {"role": "user", "parts": \[{ "text": "Given the following educational text, identify 3-7 key concepts or keywords that are central to its meaning. List them as a comma-separated string, e.g., 'concept1, concept2, keyword3'.\\n\\nText: {chunk\['content'\]}\\n\\nKeywords:" }\]}  
           \],  
           "generationConfig": {  
               "responseMimeType": "text/plain" // Or "application/json" with schema for structured keywords  
           }  
       }

     * Parse the LLM's response and add the keywords as keywords (list of strings) to the chunk's metadata.  
  4. **Pedagogical Metadata (Optional but Powerful):** Consider adding:  
     * difficulty\_level: "beginner", "intermediate", "advanced" (could be LLM-inferred or manually tagged).  
     * learning\_objectives: Specific learning outcomes addressed by the chunk.  
     * relevant\_question\_types: "definition", "process", "example", "comparison".  
     * source\_url, page\_range, author (if applicable).  
* **Output:** A list of fully augmented ChunkData objects: \[ { "unique\_id": "uuid1", "content": "...", "metadata": { "chapter\_number": 1, "chapter\_title": "...", "section\_title": "...", "keywords": \["...", "..."\], ... } }, ... \]

#### **Phase 2: Indexing for Retrieval (Hybrid Search Setup)**

This phase prepares your augmented chunks for efficient searching.

**Step 2.1: Generate Embeddings for all Chunks**

* **Action:** Create a vector representation for the content of every augmented chunk.  
* **Method:** Use the *same* sentence-transformers model that you used for semantic chunking (Step 1.2) to encode the content of each ChunkData object.  
* **Output:** Each ChunkData object now has an embedding field: \[ { "unique\_id": "uuid1", "content": "...", "embedding": \[...\], "metadata": {...} }, ... \]

**Step 2.2: Initialize and Populate Vector Database (for Semantic Search)**

* **Action:** Store your chunk embeddings and associated metadata in a vector database for fast similarity search.  
* **Method:**  
  1. **Initialize DB Client:** Connect to your chosen vector database (e.g., ChromaDB.PersistentClient(path="/path/to/db") or cloud client).  
  2. **Create Collection/Index:** Define a collection/index for your chunks.  
  3. **Add Data:** Iterate through your ChunkData list. For each chunk, add its unique\_id, embedding, content, and its metadata dictionary to the vector database.  
     * *Important:* Ensure the metadata is correctly stored and retrievable alongside the vector.  
* **Output:** A populated vector database ready for semantic queries.

**Step 2.3: Initialize and Populate Keyword Search Index (for BM25)**

* **Action:** Create a separate index for keyword-based search, focusing on term frequency.  
* **Method:**  
  1. **Prepare Documents:** Create a list of tokenized content for each chunk. (e.g., \[chunk\['content'\].split() for chunk in augmented\_chunks\]).  
  2. **Initialize BM25:** Use BM25Okapi(documents) from rank\_bm25.  
  3. **Store Mapping:** Keep a mapping of original unique\_id to the index in the BM25 object's document list.  
* **Output:** A populated BM25 object (or other full-text search index) ready for keyword queries.

#### **Phase 3: Query Processing and Retrieval (Hybrid Search Execution)**

This phase describes the real-time process when a student asks a question.

**Step 3.1: Receive User Query**

* **Action:** Capture the student's question.  
* **Method:** Get the raw text string of the user's query.

**Step 3.2: Perform Parallel Searches (Semantic & Keyword)**

* **Action:** Query both your vector database and keyword index simultaneously.  
* **Method:**  
  1. **Embed User Query:** Use the *same* sentence-transformers model to generate an embedding for the user's query.  
  2. **Semantic Search:**  
     * Query the vector database with the user's query embedding.  
     * Retrieve the Top-K\_semantic most similar chunks (e.g., K\_semantic=15).  
     * Store results as a list of (chunk\_id, semantic\_score).  
  3. **Keyword Search:**  
     * Tokenize the user's query: query\_tokens \= query\_text.split().  
     * Use the BM25 model (bm25.get\_scores(query\_tokens)) to get scores for all documents.  
     * Sort and retrieve the Top-K\_keyword most relevant chunks (e.g., K\_keyword=15).  
     * Store results as a list of (chunk\_id, keyword\_score).  
* **Output:** Two ranked lists of (chunk\_id, score) pairs, one from semantic, one from keyword search.

**Step 3.3: Fuse and Re-rank Retrieved Chunks (Reciprocal Rank Fusion \- RRF)**

* **Action:** Combine the results from both searches into a single, robustly ranked list.  
* **Method (Reciprocal Rank Fusion \- RRF):**  
  1. **Initialize RRF Scores:** Create a dictionary rrf\_scores \= {} to store combined scores for all unique chunk\_ids found in either search.  
  2. **Process Semantic Ranks:** For each (chunk\_id, semantic\_score) in your semantic results, calculate its rank (1 for top, 2 for second, etc.). Add 1 / (k \+ rank) to rrf\_scores\[chunk\_id\]. (k is a constant, typically 60).  
  3. **Process Keyword Ranks:** Similarly, for each (chunk\_id, keyword\_score) in your keyword results, calculate its rank. Add 1 / (k \+ rank) to rrf\_scores\[chunk\_id\].  
  4. **Sort by RRF Score:** Sort all chunk\_ids in rrf\_scores in descending order of their combined score.  
  5. **Select Final Top N:** Choose the Top N (e.g., N=5 to 10\) chunk\_ids from this final ranked list.  
* **Output:** A list of unique\_ids of the most relevant chunks.

**Step 3.4: Prepare Final Context for Generative LLM**

* **Action:** Retrieve the full content and relevant metadata for the Top N chosen chunks and format them for the LLM.  
* **Method:**  
  1. **Fetch Full Chunks:** Retrieve the complete ChunkData objects (including content and metadata) for the Top N unique\_ids from your storage (e.g., by querying the vector database for the full object).  
  2. **Construct LLM Prompt:** Create a structured prompt that clearly separates the instructions, the user's query, and the retrieved context.  
     * **Example Prompt Structure for LLM:**  
       "You are an expert AI teacher. Answer the student's question accurately and thoroughly, using \*only\* the information provided in the 'Context from Curriculum' section. If the answer is not directly available in the context, state that you don't have enough information to answer based on the provided materials. Be clear, concise, and explain concepts simply. Cite the Chapter and Section for each piece of information you use from the context."

       "\*\*Student Question:\*\* {user\_query}"

       "\*\*Context from Curriculum:\*\*"  
       "--- Retrieved Chunk 1 (Chapter {chunk1\_metadata\['chapter\_number'\]}: {chunk1\_metadata\['chapter\_title'\]}, Section: {chunk1\_metadata\['section\_title'\]}):"  
       "{chunk1\_content}"  
       "--- Retrieved Chunk 2 (Chapter {chunk2\_metadata\['chapter\_number'\]}: {chunk2\_metadata\['chapter\_title'\]}, Section: {chunk2\_metadata\['section\_title'\]}):"  
       "{chunk2\_content}"  
       \# ... repeat for all Top N chunks ...

       "\*\*Your Answer:\*\*"

* **Output:** A complete, well-formed prompt string ready to be sent to your generative LLM API.

This detailed plan provides the functional blueprint for setting up your RAG pipeline. Remember to test each phase rigorously and adjust parameters (like semantic\_threshold, K\_semantic, K\_keyword, N for RRF, and the RRF constant k) based on your specific dataset and desired performance.