# book_ai/views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from .rag_core import get_answer, initialize_retriever
from .models import Document, Chunk
from .forms import DocumentUploadForm
from .rag_utils import HierarchicalChunker, MetadataEnricher
import json

def home(request):
    """Home view with question-answering functionality."""
    answer = None
    query = None
    
    document_count = Document.objects.count()
    chunk_count = Chunk.objects.count()
    
    if request.method == 'POST':
        query = request.POST.get('query')
        if query:
            answer = get_answer(query)
    
    return render(request, 'book_ai/home.html', {
        'answer': answer, 
        'query': query,
        'document_count': document_count,
        'chunk_count': chunk_count
    })
 
def upload_document(request):
    """
    View to handle document upload, chunking, and embedding.
    Shows detailed progress information and handles errors gracefully.
    """
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Save the document
                document = form.save()
                messages.info(request, f"File '{document.name}' uploaded successfully. Processing started...")
                
                # Process the document with our custom RAG pipeline
                chunker = HierarchicalChunker()
                enricher = MetadataEnricher()
                
                # Get the file object from the form
                file_obj = request.FILES['file']
                
                # Check if file is supported
                file_extension = document.name.split('.')[-1].lower()
                if file_extension not in ['pdf', 'txt', 'md', 'docx']:
                    document.delete()
                    messages.error(request, f"Unsupported file format: .{file_extension}. Please upload PDF, TXT, MD, or DOCX files.")
                    return redirect('book_ai:upload_document')
                
                # Extract chunks using hierarchical chunking
                try:
                    messages.info(request, "Extracting text and creating semantic chunks...")
                    chunks = chunker.process_document(file_obj)
                except Exception as e:
                    document.delete()
                    messages.error(request, f"Error during document chunking: {str(e)}")
                    return redirect('book_ai:upload_document')
                
                # No valid chunks found
                if not chunks:
                    document.delete()
                    messages.error(request, "Could not extract any content from the document. The document might be empty or in an unsupported format.")
                    return redirect('book_ai:upload_document')
                
                # Enrich the chunks with metadata
                try:
                    messages.info(request, "Enriching chunks with metadata...")
                    enriched_chunks = enricher.enrich_chunks(chunks)
                except Exception as e:
                    document.delete()
                    messages.error(request, f"Error during metadata enrichment: {str(e)}")
                    return redirect('book_ai:upload_document')
                
                # Create Chunk entries
                chunk_objs = []
                for chunk in enriched_chunks:
                    # Store the text and metadata
                    chunk_objs.append(
                        Chunk(
                            document=document, 
                            text_content=chunk['text_content'],
                            metadata=chunk['metadata'],
                            # We'll compute embeddings in the retriever
                            embedding=None
                        )
                    )
                
                # Save all chunks to database
                try:
                    Chunk.objects.bulk_create(chunk_objs)
                except Exception as e:
                    document.delete()
                    messages.error(request, f"Error saving chunks to database: {str(e)}")
                    return redirect('book_ai:upload_document')
                
                # Update total content in document
                document.content = "\n\n".join(chunk["text_content"] for chunk in chunks)
                document.save()
                
                # Re-initialize the retriever with new chunks
                try:
                    messages.info(request, "Initializing retrieval system with new document...")
                    initialize_retriever()
                except Exception as e:
                    messages.warning(request, f"Document saved but there was an error initializing the retrieval system: {str(e)}")
                    return redirect('book_ai:home')
                
                messages.success(request, f"Document processed successfully! Created {len(chunk_objs)} semantic chunks for retrieval.")
                return redirect('book_ai:home')
                
            except Exception as e:
                # Catch-all exception handler
                try:
                    # Try to clean up if document was created
                    if 'document' in locals() and document:
                        document.delete()
                except:
                    pass
                    
                messages.error(request, f"Unexpected error processing document: {str(e)}")
                return redirect('book_ai:upload_document')
    else:
        form = DocumentUploadForm()
    
    return render(request, 'book_ai/upload.html', {'form': form, 'supported_formats': ['PDF', 'TXT', 'MD', 'DOCX']})