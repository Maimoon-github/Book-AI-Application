from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from .models import Book, BookChapter, Question, UserProfile
from .forms import SignUpForm, LoginForm, UserProfileForm, UserEditForm
import os
import tempfile
from .utils.pdf_processor import PDFProcessor
# Try to import enhanced processor, fall back to basic if dependencies missing
try:
    from .utils.enhanced_pdf_processor import EnhancedPDFProcessor
    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError:
    ENHANCED_PROCESSOR_AVAILABLE = False
# Using the improved RAG implementation for better chunking
from .utils.improved_book_teaching_rag import BookTeachingRAG
from django.views.decorators.csrf import csrf_exempt
import json
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                return redirect('home')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = LoginForm()
    return render(request, 'book_ai/login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out successfully!")
    return redirect('login')

def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Create user profile
            UserProfile.objects.create(user=user)
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            messages.success(request, f"Account created successfully. Welcome, {username}!")
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'book_ai/signup.html', {'form': form})

@login_required
def profile_view(request):
    # Check if profile exists, if not create one
    try:
        profile = request.user.profile
    except:
        profile = UserProfile.objects.create(user=request.user)
        
    if request.method == 'POST':
        user_form = UserEditForm(request.POST, instance=request.user)
        profile_form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile was successfully updated!')
            return redirect('profile')
    else:
        user_form = UserEditForm(instance=request.user)
        profile_form = UserProfileForm(instance=profile)
    
    return render(request, 'book_ai/profile.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })

@login_required
def home(request):
    # If a default API key exists in session, retrieve it
    groq_api_key = request.session.get('groq_api_key', '')
    preferred_model = request.session.get('preferred_model', 'llama-3.3-70b-versatile')
    
    books = Book.objects.all().order_by('-uploaded_at')
    return render(request, 'book_ai/home.html', {
        'books': books, 
        'groq_api_key': groq_api_key,
        'preferred_model': preferred_model
    })

@csrf_exempt
@login_required
def upload_book(request):
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        pdf_file = request.FILES['pdf_file']
        custom_name = request.POST.get('book_name', '').strip()
        
        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB in bytes
        if pdf_file.size > max_size:
            return JsonResponse({
                'status': 'error',
                'message': 'File size exceeds 50MB limit. Please upload a smaller file.'
            })
        
        # Validate file type
        if not pdf_file.name.lower().endswith('.pdf'):
            return JsonResponse({
                'status': 'error',
                'message': 'Only PDF files are supported.'
            })
          # Save the file temporarily
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                for chunk in pdf_file.chunks():
                    tmp_file.write(chunk)
                temp_file_path = tmp_file.name
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to save uploaded file: {str(e)}'            })
        
        try:
            # Use enhanced processor if available, otherwise fall back to basic
            if ENHANCED_PROCESSOR_AVAILABLE:
                processor = EnhancedPDFProcessor(temp_file_path)
            else:
                processor = PDFProcessor(temp_file_path)
            result = processor.process_pdf()
            
            if result:
                # Create Book instance with custom name
                book = Book.objects.create(
                    title=result['filename'],
                    custom_name=custom_name if custom_name else result['filename'],
                    file_path=temp_file_path,
                    page_count=result['page_count']
                )
                
                # Create BookChapter instances
                for chunk in result['hierarchical_chunks']:
                    chapter = BookChapter.objects.create(
                        book=book,
                        title=chunk['title'],
                        level=chunk['level'],
                        start_page=chunk['start_page'],
                        end_page=chunk['end_page'],
                        content=chunk['content'],
                        parent_id=chunk.get('parent_id')
                    )
                
                return JsonResponse({
                    'status': 'success',
                    'book_id': book.id,
                    'message': f'Successfully processed {book.get_display_name()}'
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to process PDF'
                })
                
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    })

@login_required
def view_book(request, book_id):
    book = Book.objects.get(id=book_id)
    chapters = book.chapters.filter(parent=None).prefetch_related('children')
    
    # Build hierarchical TOC structure for better navigation
    toc_structure = build_toc_structure(chapters)
    toc_stats = calculate_toc_stats(book.chapters.all())
    
    return render(request, 'book_ai/view_book.html', {
        'book': book,
        'chapters': chapters,
        'toc_structure': toc_structure,
        'toc_stats': toc_stats
    })

def build_toc_structure(chapters):
    """Build a hierarchical TOC structure for enhanced navigation"""
    toc_tree = []
    
    def build_children(parent_chapters):
        children = []
        for chapter in parent_chapters:
            child_data = {
                'id': chapter.id,
                'title': chapter.title,
                'level': chapter.level,
                'start_page': chapter.start_page,
                'end_page': chapter.end_page,
                'page_count': (chapter.end_page - chapter.start_page + 1) if chapter.end_page else 1,
                'children': build_children(chapter.children.all()) if hasattr(chapter, 'children') else []
            }
            children.append(child_data)
        return children
    
    return build_children(chapters)

def calculate_toc_stats(all_chapters):
    """Calculate TOC statistics for analysis"""
    if not all_chapters:
        return {}
    
    levels = [chapter.level for chapter in all_chapters if chapter.level]
    total_pages = sum((chapter.end_page - chapter.start_page + 1) 
                     for chapter in all_chapters 
                     if chapter.start_page and chapter.end_page)
    
    return {
        'total_chapters': len(all_chapters),
        'max_depth': max(levels) if levels else 0,
        'total_pages_covered': total_pages,
        'avg_chapter_length': total_pages / len(all_chapters) if all_chapters else 0,
        'chapter_distribution': {f'level_{level}': levels.count(level) for level in set(levels)} if levels else {}
    }

@csrf_exempt
def ask_question(request, book_id):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question')
        chapter_id = data.get('chapter_id')
        
        book = Book.objects.get(id=book_id)
        
        # Initialize RAG system
        rag = BookTeachingRAG()
        
        # Get chapter content if specified
        chapter_filter = None
        if chapter_id:
            chapter = BookChapter.objects.get(id=chapter_id)
            chapter_filter = chapter.title
        
        # Get previous messages from session
        messages_history = request.session.get('teaching_messages', [])
          # Generate response
        try:
            # Try to get API key in the following order:
            # 1. Book-specific API key
            # 2. Session API key
            # 3. Settings API key (fallback)
            groq_api_key = book.groq_api_key if book.groq_api_key else request.session.get('groq_api_key', settings.GROQ_API_KEY)
            model_name = book.preferred_model if book.preferred_model else request.session.get('preferred_model', 'llama-3.3-70b-versatile')
            
            rag.setup_groq_model(groq_api_key, model_name)
            
            # Index book content if not already done
            if 'book_indexed' not in request.session:
                chapters = book.chapters.all()
                book_chunks = [{
                    'title': chapter.title,
                    'level': chapter.level,
                    'start_page': chapter.start_page,
                    'end_page': chapter.end_page,
                    'content': chapter.content
                } for chapter in chapters]
                rag.index_book_content(book_chunks)
                request.session['book_indexed'] = True
            
            # Get response
            response = rag.teach_topic(
                question,
                messages_history,
                chapter_filter,
                f"session_{request.session.session_key}"
            )
            
            # Update messages history in session
            messages_history.append({"role": "user", "content": question})
            messages_history.append({
                "role": "assistant",
                "content": response["response"].content,
                "sources": response["sources"]
            })
            request.session['teaching_messages'] = messages_history
            
            # Record question if chapter specified
            if chapter_id:
                question_obj, created = Question.objects.get_or_create(
                    chapter_id=chapter_id,
                    text=question,
                    defaults={'frequency': 1}
                )
                if not created:
                    question_obj.frequency += 1
                    question_obj.save()
            
            return JsonResponse({
                'status': 'success',
                'response': response["response"].content,
                'sources': response["sources"]
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    })

def get_frequent_questions(request, chapter_id):
    chapter = BookChapter.objects.get(id=chapter_id)
    questions = chapter.questions.order_by('-frequency')[:5]
    return JsonResponse({
        'questions': [{
            'text': q.text,
            'frequency': q.frequency
        } for q in questions]
    })

def clear_chat(request, book_id):
    if 'teaching_messages' in request.session:
        del request.session['teaching_messages']
    return redirect('view_book', book_id=book_id)

@csrf_exempt
def update_api_settings(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        groq_api_key = data.get('groq_api_key', '')
        model_name = data.get('model_name', 'llama-3.3-70b-versatile')
        
        # Store in session for use across the site
        request.session['groq_api_key'] = groq_api_key
        request.session['preferred_model'] = model_name
        
        # If we're working with a specific book, update its settings too
        book_id = data.get('book_id')
        if book_id:
            try:
                book = Book.objects.get(id=book_id)
                book.groq_api_key = groq_api_key
                book.preferred_model = model_name
                book.save()
            except Book.DoesNotExist:
                pass
        
        return JsonResponse({
            'status': 'success',
            'message': 'API settings updated successfully'
        })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@csrf_exempt
def rate_response(request):
    """Endpoint to rate an AI response as useful or not useful"""
    if request.method == 'POST':
        data = json.loads(request.body)
        message_id = data.get('message_id')
        rating = data.get('rating')  # 'useful' or 'not-useful'
        book_id = data.get('book_id')
        
        if not all([message_id, rating, book_id]):
            return JsonResponse({
                'status': 'error',
                'message': 'Missing required fields'
            })
            
        try:
            book = Book.objects.get(id=book_id)
            
            # Create or update rating
            from .models import ResponseRating
            
            rating_obj, created = ResponseRating.objects.get_or_create(
                book=book,
                message_id=message_id,
                defaults={'rating': rating}
            )
            
            if not created:
                rating_obj.rating = rating
                rating_obj.save()
                
            return JsonResponse({
                'status': 'success',
                'message': 'Rating saved successfully'
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
            
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })
    
@csrf_exempt
def submit_feedback(request):
    """Endpoint to submit detailed feedback for an AI response"""
    if request.method == 'POST':
        data = json.loads(request.body)
        message_id = data.get('message_id')
        feedback = data.get('feedback')
        book_id = data.get('book_id')
        
        if not all([message_id, feedback, book_id]):
            return JsonResponse({
                'status': 'error',
                'message': 'Missing required fields'
            })
            
        try:
            book = Book.objects.get(id=book_id)
            
            # Find existing rating if any
            from .models import ResponseRating, ResponseFeedback
            
            rating_obj = None
            try:
                rating_obj = ResponseRating.objects.get(
                    book=book,
                    message_id=message_id
                )
            except ResponseRating.DoesNotExist:
                pass
                
            # Create feedback
            feedback_obj = ResponseFeedback.objects.create(
                rating=rating_obj,
                book=book,
                message_id=message_id,
                feedback_text=feedback
            )
                
            return JsonResponse({
                'status': 'success',
                'message': 'Feedback saved successfully'
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
            
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@csrf_exempt
def store_question(request, book_id):
    """Endpoint to store questions for analytics and frequency tracking"""
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question')
        chapter_id = data.get('chapter_id')
        
        if not question:
            return JsonResponse({
                'status': 'error',
                'message': 'Missing required fields'
            })
            
        # If chapter_id is provided, update question frequency
        if chapter_id:
            try:
                chapter = BookChapter.objects.get(id=chapter_id)
                question_obj, created = Question.objects.get_or_create(
                    chapter=chapter,
                    text=question,
                    defaults={'frequency': 1}
                )
                
                if not created:
                    question_obj.frequency += 1
                    question_obj.save()
                    
                return JsonResponse({
                    'status': 'success',
                    'message': 'Question stored successfully'
                })
                
            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'message': str(e)
                })
        
        # If no chapter, just return success (we still track it in session)
        return JsonResponse({
            'status': 'success',
            'message': 'Question recorded in session'
        })
            
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@csrf_exempt
@login_required
def delete_book(request, book_id):
    """Delete a book and all its associated data"""
    if request.method == 'POST':
        try:
            book = Book.objects.get(id=book_id)
            
            # Delete the physical file if it exists
            if book.file_path and os.path.exists(book.file_path):
                os.unlink(book.file_path)
            
            # Delete the book (this will cascade delete chapters due to foreign key)
            book.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Book deleted successfully'
            })
        except Book.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Book not found'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@csrf_exempt
@login_required
def rename_book(request, book_id):
    """Rename a book"""
    if request.method == 'POST':
        try:
            book = Book.objects.get(id=book_id)
            new_name = request.POST.get('new_name', '').strip()
            
            if not new_name:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Book name cannot be empty'
                })
            
            book.custom_name = new_name
            book.save()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Book renamed successfully',
                'new_name': new_name
            })
        except Book.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Book not found'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@login_required
def download_book(request, book_id):
    """Download the original PDF file"""
    try:
        book = Book.objects.get(id=book_id)
        
        if not book.file_path or not os.path.exists(book.file_path):
            return HttpResponse('File not found', status=404)
        
        with open(book.file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{book.get_display_name()}.pdf"'
            return response
            
    except Book.DoesNotExist:
        return HttpResponse('Book not found', status=404)

@csrf_exempt
@login_required
def delete_profile_picture(request):
    """Delete user's profile picture"""
    if request.method == 'POST':
        try:
            profile = request.user.profile
            
            # Delete the physical file if it exists
            if profile.profile_picture and hasattr(profile.profile_picture, 'path'):
                if os.path.exists(profile.profile_picture.path):
                    os.unlink(profile.profile_picture.path)
            
            # Clear the profile picture field
            profile.profile_picture.delete(save=True)
            
            return JsonResponse({
                'status': 'success',
                'message': 'Profile picture deleted successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })
