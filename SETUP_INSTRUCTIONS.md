# Book AI Application - Setup Instructions

## Quick Setup Guide

### Prerequisites
- Python 3.11+ (Conda environment recommended)
- Groq API key for AI features

### Installation Steps

1. **Activate your conda environment**
   ```bash
   conda activate book_ai
   ```

2. **Install dependencies**
   ```bash
   # For Django web app
   cd rag_book_ai
   pip install -r requirements.txt
   
   # For Streamlit app
   cd ../Book\ App
   pip install -r requirements.txt
   
   # Or install all dependencies
   cd ..
   pip install -r requirements-complete.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file in root directory
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   echo "SECRET_KEY=your_django_secret_key_here" >> .env
   echo "DEBUG=True" >> .env
   ```

4. **Initialize Django database**
   ```bash
   cd rag_book_ai
   python manage.py makemigrations
   python manage.py migrate
   python manage.py createsuperuser  # Optional
   ```

### Running the Applications

#### Django Web App
```bash
cd rag_book_ai
python manage.py runserver
```
Access at: http://localhost:8000

#### Streamlit App
```bash
cd Book\ App
streamlit run book_ai.py
```
Access at: http://localhost:8501

### Key Features
- **PDF Processing**: Upload and analyze PDF books
- **AI Chat**: Interactive Q&A with book content
- **Chapter Navigation**: Organized content by chapters
- **User Profiles**: Personal accounts and preferences
- **Enhanced Processing**: Improved PDF parsing when dependencies available

### Troubleshooting
- Ensure all dependencies are installed in your conda environment
- Check that GROQ_API_KEY is set correctly
- For import errors, install missing packages: `pip install <package_name>`
- Large PDF files may take longer to process

### Architecture
- **Django App**: Full-featured web application with user management
- **Streamlit App**: Quick prototyping and testing interface
- **Shared Utils**: Common PDF processing and RAG functionality
