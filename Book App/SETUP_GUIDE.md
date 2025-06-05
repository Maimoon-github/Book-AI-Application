# Book AI Application - Setup and Usage Guide

## 🚀 Quick Start

Your Book AI Application has been successfully improved and is ready to use! Here's how to get started:

## ✅ Completed Improvements

1. **PyTorch Warning Suppression** - No more harmless PyTorch warnings cluttering the output
2. **Sliding Window RAG Chunking** - Better context continuity with 400-word windows and 50-word overlap
3. **Enhanced Export Options** - Now supports CSV, PDF text, and Markdown formats
4. **Pandas Warning Fix** - Eliminated SettingWithCopyWarning using proper DataFrame operations
5. **Code Quality** - Fixed all syntax and indentation errors

## 🛠️ Setup Instructions

### Option 1: Using the existing environment
If streamlit is already installed in your environment:
```bash
cd "d:\maimoon\Vs Code\Book-AI-Application\Book App"
streamlit run book_ai.py
```

### Option 2: Install dependencies
If you need to install packages:
```bash
cd "d:\maimoon\Vs Code\Book-AI-Application\Book App"
pip install -r requirements.txt
streamlit run book_ai.py
```

### Option 3: Using Python module syntax
```bash
cd "d:\maimoon\Vs Code\Book-AI-Application\Book App"
python -m streamlit run book_ai.py
```

## 🎯 Features Overview

### 📖 Document Processing
- **Smart Chapter Detection**: Uses bookmarks or regex patterns to identify chapters
- **Hierarchical Structure**: Builds parent-child relationships between sections
- **Page Mapping**: Accurate page number tracking for all content

### 🔍 Advanced Search
- **Full-text Search**: Find content across the entire document
- **Chapter Filtering**: Focus searches on specific chapters
- **Context Highlighting**: See matches in context

### 🤖 AI Teaching Assistant
- **Groq API Integration**: Powered by advanced language models
- **RAG (Retrieval Augmented Generation)**: Contextual responses based on book content
- **Memory**: Maintains conversation history for coherent interactions
- **Source References**: Shows which parts of the book were used for responses

### 📊 Export Options
- **CSV**: Structured data for spreadsheet analysis
- **Markdown**: Documentation-friendly format
- **PDF Text**: Formatted text file with document structure

## 🔧 Configuration

### API Setup
1. Get a free Groq API key from: https://console.groq.com/
2. Enter the API key in the sidebar when running the application
3. Choose your preferred AI model from the dropdown

### Supported Models
- llama-3.3-70b-versatile (recommended)
- llama-3.1-8b-instant
- gemma2-9b-it
- mixtral-8x7b-32768

## 📚 Usage Tips

### For Best Results
1. **Upload Clear PDFs**: Ensure your PDF has readable text (not just images)
2. **Use Descriptive Questions**: Ask specific questions about concepts, themes, or chapters
3. **Explore Hierarchically**: Start with broad topics, then drill down into specifics
4. **Reference Sources**: Check the sources provided with AI responses

### Example Questions for AI Teacher
- "Explain the main concept in Chapter 3"
- "How does this relate to the previous chapter?"
- "Can you give me practical examples of this theory?"
- "What are the key takeaways from this section?"
- "How can I apply this knowledge in real life?"

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Make sure all packages from requirements.txt are installed
2. **PDF Processing Fails**: Ensure the PDF contains text (not just scanned images)
3. **AI Responses Not Working**: Check that your Groq API key is valid and entered correctly
4. **Streamlit Won't Start**: Try using `python -m streamlit run book_ai.py`

### Performance Tips
- For large books (>500 pages), processing may take a few minutes
- The first AI response may be slower as the system initializes
- Clear chat history periodically for better performance

## 📝 File Structure

```
Book App/
├── book_ai.py              # Main application
├── requirements.txt        # Python dependencies
├── verify_improvements.py  # Verification script
├── test_startup.py        # Startup test script
└── Books/                 # Sample PDF files
```

## 🎉 You're All Set!

Your Book AI Application is now ready to transform how you interact with books. Upload a PDF, ask questions, and let the AI help you learn more effectively!

For support or questions, check the code comments or run the verification script to ensure everything is working correctly.
