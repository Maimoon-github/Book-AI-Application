from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import json
import os
import uuid
from .models import Book, Chapter, ChatSession, ChatMessage

# Import the PDF processing functionality
try:
    from .utils import PDFProcessor, BookTeachingRAG, SUPPORTED_MODELS, PDF_PROCESSING_AVAILABLE
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    SUPPORTED_MODELS = []

def home(request):
    """Main dashboard view"""
    books = Book.objects.all().order_by('-uploaded_at')
    context = {
        'books': books,
        'pdf_processing_available': PDF_PROCESSING_AVAILABLE,
        'supported_models': SUPPORTED_MODELS if PDF_PROCESSING_AVAILABLE else []
    }
    return render(request, 'application/home.html', context)

def upload_book(request):
    """Handle book upload and processing"""
    if request.method == 'POST':
        if 'pdf_file' not in request.FILES:
            messages.error(request, 'No file uploaded.')
            return redirect('home')
        
        pdf_file = request.FILES['pdf_file']
        
        if not pdf_file.name.endswith('.pdf'):
            messages.error(request, 'Please upload a PDF file.')
            return redirect('home')
        
        # Save the book
        book = Book.objects.create(
            title=pdf_file.name.replace('.pdf', ''),
            filename=pdf_file.name,
            file_path=pdf_file,
            uploaded_by=request.user if request.user.is_authenticated else None,
            processing_status='pending'
        )
        
        messages.success(request, f'Book "{book.title}" uploaded successfully. Processing...')
        
        # Process the PDF in the background (in a real application, use Celery)
        try:
            process_pdf_book(book)
            messages.success(request, f'Book "{book.title}" processed successfully!')
        except Exception as e:
            book.processing_status = 'failed'
            book.save()
            messages.error(request, f'Failed to process book: {str(e)}')
        
        return redirect('book_detail', book_id=book.id)
    
    return render(request, 'application/upload.html')

def process_pdf_book(book):
    """Process a PDF book and extract chapters"""
    if not PDF_PROCESSING_AVAILABLE:
        raise Exception("PDF processing libraries not available")
    
    book.processing_status = 'processing'
    book.save()
    
    # Get the file path
    file_path = book.file_path.path
    
    # Process the PDF
    processor = PDFProcessor(file_path)
    result = processor.process_pdf()
    
    if not result:
        raise Exception("Failed to process PDF")
    
    # Update book info
    book.page_count = result['page_count']
    book.processing_status = 'completed'
    book.save()
    
    # Create chapters
    for chunk in result['hierarchical_chunks']:
        parent_chapter = None
        if chunk.get('parent_id'):
            parent_chapter = Chapter.objects.filter(
                book=book, 
                chapter_id=chunk['parent_id']
            ).first()
        
        Chapter.objects.create(
            book=book,
            title=chunk['title'],
            level=chunk['level'],
            start_page=chunk['start_page'],
            end_page=chunk['end_page'],
            content=chunk['content'],
            parent=parent_chapter,
            chapter_id=chunk['id']
        )

def book_detail(request, book_id):
    """Display book details and chapters"""
    book = get_object_or_404(Book, id=book_id)
    chapters = book.chapters.filter(parent=None)  # Root chapters only
    
    context = {
        'book': book,
        'chapters': chapters,
        'pdf_processing_available': PDF_PROCESSING_AVAILABLE
    }
    return render(request, 'application/book_detail.html', context)

def chapter_content(request, chapter_id):
    """Get chapter content via AJAX"""
    chapter = get_object_or_404(Chapter, id=chapter_id)
    
    data = {
        'title': chapter.title,
        'content': chapter.content,
        'start_page': chapter.start_page + 1,  # Convert to 1-indexed
        'end_page': chapter.end_page + 1,
        'level': chapter.level,
        'children': [
            {
                'id': child.id,
                'title': child.title,
                'start_page': child.start_page + 1,
                'end_page': child.end_page + 1
            }
            for child in chapter.children.all()
        ]
    }
    
    return JsonResponse(data)

def search_content(request, book_id):
    """Search within book content"""
    book = get_object_or_404(Book, id=book_id)
    query = request.GET.get('q', '').strip()
    
    if not query:
        return JsonResponse({'results': []})
    
    # Search in chapter content
    chapters = book.chapters.filter(content__icontains=query)
    
    results = []
    for chapter in chapters:
        # Find the context around the search term
        content = chapter.content.lower()
        query_lower = query.lower()
        index = content.find(query_lower)
        
        if index != -1:
            start = max(0, index - 100)
            end = min(len(content), index + len(query) + 100)
            context = chapter.content[start:end]
            
            results.append({
                'chapter_id': chapter.id,
                'title': chapter.title,
                'context': context,
                'start_page': chapter.start_page + 1,
                'end_page': chapter.end_page + 1
            })
    
    return JsonResponse({'results': results})

def chat_interface(request, book_id):
    """AI chat interface for the book"""
    book = get_object_or_404(Book, id=book_id)
    
    if not PDF_PROCESSING_AVAILABLE:
        messages.error(request, 'AI features are not available. Please install required packages.')
        return redirect('book_detail', book_id=book_id)
    
    # Get or create chat session
    session_id = request.session.get(f'chat_session_{book_id}')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session[f'chat_session_{book_id}'] = session_id
    
    chat_session, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={
            'book': book,
            'user': request.user if request.user.is_authenticated else None
        }
    )
    
    messages_list = chat_session.messages.all()
    
    context = {
        'book': book,
        'chat_session': chat_session,
        'messages': messages_list,
        'supported_models': SUPPORTED_MODELS
    }
    
    return render(request, 'application/chat.html', context)

@csrf_exempt
def send_message(request, book_id):
    """Handle AI chat messages"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    book = get_object_or_404(Book, id=book_id)
    
    if not PDF_PROCESSING_AVAILABLE:
        return JsonResponse({'error': 'AI features not available'}, status=400)
    
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        api_key = data.get('api_key', '').strip()
        model = data.get('model', SUPPORTED_MODELS[0])
        chapter_filter = data.get('chapter_filter')
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        if not api_key:
            return JsonResponse({'error': 'Groq API key is required'}, status=400)
        
        # Get chat session
        session_id = request.session.get(f'chat_session_{book_id}')
        if not session_id:
            return JsonResponse({'error': 'No chat session found'}, status=400)
        
        chat_session = get_object_or_404(ChatSession, session_id=session_id)
        
        # Save user message
        user_message = ChatMessage.objects.create(
            session=chat_session,
            message_type='human',
            content=message
        )
        
        # Initialize RAG system
        rag_system = BookTeachingRAG()
        rag_system.setup_groq_model(api_key, model)
        
        # Get book chunks for RAG
        book_chunks = []
        chapters = book.chapters.all()
        for chapter in chapters:
            book_chunks.append({
                'title': chapter.title,
                'content': chapter.content,
                'start_page': chapter.start_page,
                'end_page': chapter.end_page,
                'level': chapter.level
            })
        
        # Index content
        rag_system.index_book_content(book_chunks)
        
        # Get message history
        previous_messages = []
        for msg in chat_session.messages.filter(id__lt=user_message.id):
            if msg.message_type == 'human':
                previous_messages.append({"role": "user", "content": msg.content})
            else:
                previous_messages.append({"role": "assistant", "content": msg.content})
        
        # Get AI response
        response = rag_system.teach_topic(
            message, 
            previous_messages, 
            chapter_filter, 
            session_id
        )
        
        # Save AI response
        ai_message = ChatMessage.objects.create(
            session=chat_session,
            message_type='ai',
            content=response
        )
        
        return JsonResponse({
            'response': response,
            'message_id': ai_message.id
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def export_structure(request, book_id):
    """Export book structure as CSV"""
    book = get_object_or_404(Book, id=book_id)
    
    import csv
    from django.http import HttpResponse
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{book.title}_structure.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Title', 'Level', 'Start Page', 'End Page'])
    
    chapters = book.chapters.all().order_by('start_page', 'level')
    for chapter in chapters:
        writer.writerow([
            chapter.title,
            chapter.level,
            chapter.start_page + 1,  # Convert to 1-indexed
            chapter.end_page + 1
        ])
    
    return response
