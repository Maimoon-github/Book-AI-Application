# book_ai/views.py

from django.shortcuts import render
from .rag_core import get_answer
from django.shortcuts import redirect
from .models import Document, Chunk
from .forms import DocumentUploadForm
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def home(request):
    answer = None
    query = None
    if request.method == 'POST':
        query = request.POST.get('query')
        if query:
            answer = get_answer(query)
    return render(request, 'book_ai/home.html', {'answer': answer, 'query': query})
 
def upload_document(request):
    """
    View to handle document upload, chunking, and embedding.
    """
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save()
            # Read file content and store in model
            with document.file.open('r', encoding='utf-8') as f:
                text = f.read()
            document.content = text
            document.save()
            # Split into chunks
            splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = splitter.split_text(text)
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
            # Create Chunk entries
            chunk_objs = []
            for chunk_text in chunks:
                vector = embeddings.embed_documents([chunk_text])[0]
                chunk_objs.append(Chunk(document=document, text_content=chunk_text,
                                          metadata={}, embedding=vector))
            Chunk.objects.bulk_create(chunk_objs)
            return redirect('book_ai:home')
    else:
        form = DocumentUploadForm()
    return render(request, 'book_ai/upload.html', {'form': form})