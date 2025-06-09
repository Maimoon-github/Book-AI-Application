# Book AI - Robust Django RAG Application

A Retrieval-Augmented Generation (RAG) application built with Django that allows users to upload and query PDF documents. The application features a robust, fault-tolerant architecture with graceful degradation when dependencies are missing.

## Features

- **PDF Document Upload**: Upload PDF documents to extract and index their content
- **Semantic Search**: Query your documents using natural language
- **Hierarchical Chunking**: Intelligent document segmentation for better context retention
- **Robust Dependency Handling**: Falls back to simpler implementations when packages are missing
- **Hybrid Retrieval**: Combines semantic search (embeddings) and keyword search (BM25) when available

## System Requirements

- Python 3.8+
- Django 4.0+
- 2GB+ RAM (for embedding models)
- Windows, macOS, or Linux

### Windows-Specific Requirements

- Developer Mode enabled for symlinks (required by some dependencies)
  - To enable: Settings > Update & Security > For developers > Developer Mode

## Dependency Tiers

The system supports different levels of functionality based on available dependencies:

### Tier 1: Full Functionality
- FAISS for fast vector search
- Sentence Transformers for embeddings
- PyTorch for machine learning
- NLTK for text processing
- Rank BM25 for keyword search
- HuggingFace Transformers for text generation

### Tier 2: Standard Functionality
- Sentence Transformers for embeddings
- NumPy for vector operations
- Rank BM25 for keyword search
- NLTK for text processing

### Tier 3: Basic Functionality
- Sentence Transformers for embeddings
- NumPy for vector operations

### Tier 4: Minimal Functionality
- Basic text matching
- No external dependencies

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Download NLTK data:
   ```
   python download_nltk_data.py
   ```
6. Verify environment:
   ```
   python verify_env.py
   ```
7. Run migrations:
   ```
   python manage.py migrate
   ```
8. Start the development server:
   ```
   python manage.py runserver
   ```

## Environment Setup

For the best experience, install all dependencies:

```
pip install faiss-cpu~=1.7.0
pip install sentence-transformers~=2.2.0
pip install nltk~=3.8.0
pip install rank_bm25~=0.2.0
pip install PyPDF2~=3.0.0
pip install transformers~=4.36.0
pip install huggingface_hub[hf_xet]~=0.19.0
pip install tqdm~=4.65.0
pip install torch~=2.0.0
```

### Troubleshooting SSL Issues with NLTK

If you encounter SSL certificate issues when downloading NLTK data, run:

```
python download_nltk_data.py
```

This script provides a workaround for common SSL certificate issues.

## Verifying Your Environment

To check if your environment is set up correctly:

```
python verify_env.py
```

This will display which dependencies are available and provide recommendations.

## Architecture

The application uses a tiered approach to dependencies:

1. **rag_utils.py**: Contains the full-featured implementation with FAISS and all dependencies
2. **simple_retriever.py**: Uses NumPy instead of FAISS, but still requires sentence-transformers and BM25
3. **basic_retriever.py**: Minimal implementation with graceful fallbacks for nearly all dependencies

The system automatically selects the most capable implementation based on available dependencies.

## Usage

1. Start the server: `python manage.py runserver`
2. Navigate to http://127.0.0.1:8000/
3. Upload a PDF document
4. Start asking questions about your documents

## License

[MIT License](LICENSE)
