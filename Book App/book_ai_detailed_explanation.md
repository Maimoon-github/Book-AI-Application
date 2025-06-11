# Detailed Step-by-Step Explanation of `book_ai.py`

This document provides a comprehensive, step-by-step breakdown of the `book_ai.py` file. The goal is to clarify the flow, structure, and logic of the code, making it easy to port to another framework. Each section is explained in detail, including how retrieval, content processing, and component interactions work.

---

## 1. **Imports and Environment Setup**

- **Standard Libraries:**
  - `os`, `re`, `tempfile`, `warnings`: For file handling, regex, temporary files, and warning management.
  - `typing`: Type hints for better code clarity.
  - `pandas as pd`: For data manipulation and export.
- **Third-Party Libraries:**
  - `fitz` (PyMuPDF): For PDF processing.
  - `streamlit as st`: For UI (can be replaced in another framework).
- **Suppressing Warnings:**
  - PyTorch warnings are suppressed to avoid clutter in the UI.
  - Sets an environment variable to disable PyTorch version checks.

---

## 2. **RAG and AI Imports**

- **Conditional Import:**
  - Tries to import RAG (Retrieval-Augmented Generation) libraries: `langchain_groq`, `langchain_core`, `langgraph`, etc.
  - If missing, sets `RAG_AVAILABLE = False` and warns the user.

---

## 3. **Supported Models**

- A list of supported Groq models is defined for use in the AI teaching assistant.

---

## 4. **RAG Component Loader**

- `@st.cache_resource` decorator ensures that heavy resources (embedding model, ChromaDB client) are loaded only once.
- Loads the `SentenceTransformer` embedding model and initializes a ChromaDB client.

---

## 5. **Question Tracking System**

- Uses Streamlit's session state to track frequently asked questions per chapter.
- Functions:
  - `initialize_question_tracking()`: Sets up the tracking dictionary.
  - `record_question(chapter, question)`: Increments the count for a question in a chapter.
  - `get_frequent_questions(chapter, limit)`: Returns the most frequent questions for a chapter.
  - `display_frequent_questions_sidebar(...)`: Renders frequent questions in the sidebar for quick access.

---

## 6. **BookTeachingRAG Class**

### **Initialization**
- Loads RAG components (embedding model, ChromaDB client).
- Prepares placeholders for collection, Groq model, and workflow app.

### **Model Setup**
- `setup_groq_model(api_key, model_name)`: Initializes the Groq model and sets up the LangGraph workflow.

### **LangGraph Workflow**
- Defines a teaching prompt with detailed instructions for the AI.
- Teaching state includes messages, context, and sources.
- The workflow node `call_teaching_model`:
  - Adds book context to the last user message.
  - Records the question for frequency tracking.
  - Invokes the Groq model and returns the response.
- The workflow is compiled with a memory checkpoint.

### **Book Content Indexing**
- `index_book_content(book_chunks)`: Stores book chunks in the vector database (ChromaDB).
- Chunks are embedded and added to the collection.

### **Chunk Creation**
- `create_rag_chunks(book_chunks)`: Splits chapters into overlapping word windows for better retrieval.

### **Context Retrieval**
- `retrieve_context(query, chapter_filter)`: Retrieves the most relevant chunks for a query, optionally filtered by chapter.

### **Teaching Function**
- `teach_topic(user_question, messages_history, selected_chapter, thread_id)`: Main function that retrieves context, prepares state, and invokes the workflow to generate a teaching response.

---

## 7. **PDFProcessor Class**

### **Initialization**
- Stores file path, document object, filename, extension, and page character counts.

### **PDF Operations**
- `open_document()`, `close_document()`: Open/close the PDF using PyMuPDF.
- `extract_text_fitz()`: Extracts all text and records character counts per page.
- `extract_bookmarks_fitz()`: Extracts bookmarks (table of contents) from the PDF.
- `identify_chapters_regex(text)`: Uses regex to find chapter/section headings if bookmarks are missing.
- `find_page_number(char_index)`: Maps a character index in the text to a page number.

### **Chunk Processing**
- `process_bookmark_chunks(bookmarks)`: Uses bookmarks to split the document into chunks.
- `process_regex_chunks(chapter_starts, full_text)`: Uses regex-identified chapters to split the document.
- `build_hierarchical_chunks(chunks)`: Organizes chunks into a parent-child hierarchy based on levels.

### **Main Processing**
- `process_pdf()`: Orchestrates the extraction and chunking process, returning structured data about the document.

---

## 8. **Table of Contents Display**

- `display_hierarchical_toc(hierarchical_chunks)`: Renders a hierarchical table of contents and allows users to view chapter content.

---

## 9. **Export Functions**

- `export_chunks_to_csv(chunks, filename)`: Exports structure to CSV.
- `export_chunks_to_markdown(chunks, filename)`: Exports structure to Markdown.
- `export_chunks_to_pdf_text(chunks, filename)`: Exports structure to a plain text file (PDF-like format).

---

## 10. **Teaching Interface**

- `create_teaching_interface(result, api_key)`: Sets up the interactive AI teacher UI.
  - Initializes the RAG system and indexes content.
  - Lets the user select a model and chapter.
  - Handles chat history, user input, and displays AI responses with sources.
  - Integrates frequent question tracking and suggestions for further learning.

---

## 11. **Main Application Flow**

- `main()` function:
  - Sets up the Streamlit UI (title, sidebar, file uploader).
  - Handles PDF upload and processing.
  - Creates tabs for structure, search, AI teacher, and export.
  - Cleans up temporary files after processing.

---

## 12. **How Retrieval and Content Flow Work**

- **PDF is uploaded** → Processed into chunks (by bookmarks or regex) → Chunks are indexed in ChromaDB with embeddings.
- **User asks a question** → Query is embedded and used to retrieve relevant chunks from ChromaDB.
- **Relevant context is added to the prompt** → Groq model generates a teaching response using LangGraph workflow.
- **Frequent questions are tracked** and shown in the sidebar for quick access.

---

## 13. **How Components Are Tied Together**

- The PDFProcessor extracts and structures the book.
- BookTeachingRAG handles indexing, retrieval, and teaching logic.
- The main function and UI glue everything together, allowing users to interact with the system, ask questions, and export data.

---

## 14. **Code Structure Reference**

Below is the full code as it appears in `book_ai.py` for reference:

```
[Insert the full code from book_ai.py here.]
```

---

**This document should make it straightforward to port the logic to another framework, as each step and interaction is clearly explained.**
