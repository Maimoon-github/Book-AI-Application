Below is a complete Django implementation to parse PDF books, extract the table of contents (ToC), split chapters/subtopics, and store the structured data. This solution uses PyPDF2 for PDF parsing and includes models, views, and processing logic.

### 1. Models (`models.py`)
```python
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    pdf_file = models.FileField(upload_to='books/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class TableOfContent(models.Model):
    book = models.OneToOneField(Book, on_delete=models.CASCADE, related_name='toc')
    content = models.TextField()

class Chapter(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='chapters')
    title = models.CharField(max_length=200)
    order = models.PositiveIntegerField()

class SubTopic(models.Model):
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE, related_name='subtopics')
    title = models.CharField(max_length=200)
    content = models.TextField()
    order = models.PositiveIntegerField()
```

### 2. PDF Processing Utility (`pdf_utils.py`)
```python
import re
import PyPDF2
from io import BytesIO

def extract_toc_and_content(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    toc_pages = []
    content_pages = []
    
    # Extract first 5 pages as ToC (adjust page count as needed)
    for i in range(min(5, len(reader.pages))):
        toc_pages.append(reader.pages[i].extract_text())
    
    # Extract remaining pages as content
    for i in range(5, len(reader.pages)):
        content_pages.append(reader.pages[i].extract_text())
    
    return {
        'toc': "\n".join(toc_pages),
        'content': "\n".join(content_pages)
    }

def parse_toc_structure(toc_text):
    chapters = []
    lines = toc_text.split('\n')
    
    for line in lines:
        # Match chapter titles (customize regex as needed)
        if re.match(r'^(Chapter \d+|Section \d+\.\d+)', line.strip()):
            chapters.append({
                'title': line.strip(),
                'subtopics': []
            })
        # Match subtopics (indented or dotted patterns)
        elif re.match(r'^\s{4,}[A-Za-z]', line.strip()):
            if chapters:
                chapters[-1]['subtopics'].append(line.strip())
    return chapters

def map_content_to_chapters(content, toc_structure):
    chapter_content = {}
    content_lines = content.split('\n')
    current_chapter = None
    
    for line in content_lines:
        # Check if line matches a chapter title
        for idx, chap in enumerate(toc_structure):
            if chap['title'] in line:
                current_chapter = chap['title']
                chapter_content[current_chapter] = []
                break
        
        # Add content to current chapter
        if current_chapter:
            chapter_content[current_chapter].append(line)
    
    # Convert lists to strings
    return {chap: "\n".join(lines) for chap, lines in chapter_content.items()}
```

### 3. Views (`views.py`)
```python
from django.shortcuts import render, redirect
from .models import Book, TableOfContent, Chapter, SubTopic
from .pdf_utils import extract_toc_and_content, parse_toc_structure, map_content_to_chapters

def upload_book(request):
    if request.method == 'POST':
        new_book = Book.objects.create(
            title=request.FILES['pdf_file'].name,
            pdf_file=request.FILES['pdf_file']
        )
        process_book(new_book)
        return redirect('book_detail', book_id=new_book.id)
    return render(request, 'upload.html')

def process_book(book):
    # Extract raw text from PDF
    extracted = extract_toc_and_content(book.pdf_file)
    
    # Save ToC
    TableOfContent.objects.create(book=book, content=extracted['toc'])
    
    # Parse ToC structure
    toc_structure = parse_toc_structure(extracted['toc'])
    
    # Map content to chapters
    chapter_content_map = map_content_to_chapters(
        extracted['content'], 
        toc_structure
    )
    
    # Save chapters and subtopics
    for order_idx, chapter_data in enumerate(toc_structure):
        chap = Chapter.objects.create(
            book=book,
            title=chapter_data['title'],
            order=order_idx + 1
        )
        
        # Save subtopics
        for sub_idx, subtopic_title in enumerate(chapter_data['subtopics']):
            SubTopic.objects.create(
                chapter=chap,
                title=subtopic_title,
                content=chapter_content_map.get(chapter_data['title'], ''),
                order=sub_idx + 1
            )
```

### 4. Templates (`upload.html`)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Upload Book</title>
</head>
<body>
    <h1>Upload PDF Book</h1>
    <form method="post" enctype="multipart/form-data">
        {% csrf_token %}
        <input type="file" name="pdf_file" accept=".pdf">
        <button type="submit">Process Book</button>
    </form>
</body>
</html>
```

### 5. URL Configuration (`urls.py`)
```python
from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_book, name='upload_book'),
    path('book/<int:book_id>/', views.book_detail, name='book_detail'),
]
```

### Workflow Explanation:
1. **Upload PDF**:
   - User uploads PDF via web form
   - System saves file to `media/books/`

2. **PDF Processing**:
   - **ToC Extraction**: First 5 pages extracted as table of contents
   - **Content Extraction**: Remaining pages as main content
   - **Structure Parsing**:
     - Chapters detected via `Chapter X` or `Section X.X` patterns
     - Subtopics detected by indentation patterns

3. **Data Mapping**:
   - Scans content text for chapter headings
   - Groups content lines under detected chapters
   - Preserves original document order

4. **Database Storage**:
   - Book metadata
   - Raw ToC text
   - Hierarchical chapter/subtopic structure
   - Content linked to respective subtopics

### Key Features:
- **Separation of Concerns**: 
  - ToC extraction isolated from content processing
  - Structure parsing separate from content mapping
- **Flexible Pattern Matching**:
  - Regex-based detection for chapters/subtopics
  - Customizable patterns for different document styles
- **Order Preservation**:
  - Explicit `order` fields maintain document sequence
- **PDF Handling**:
  - Uses PyPDF2 for text extraction
  - Handles multi-page documents

### Installation Requirements:
```bash
pip install django PyPDF2
```

### Usage:
1. Upload PDF through `/upload/` endpoint
2. System automatically:
   - Extracts ToC and content
   - Identifies chapters/subtopics
   - Stores structured data in database
3. Access processed data via Django ORM:
```python
book = Book.objects.get(title='MyBook.pdf')
for chapter in book.chapters.all():
    print(chapter.title)
    for subtopic in chapter.subtopics.all():
        print(f"- {subtopic.title}")
        print(subtopic.content)
```

This implementation provides a foundation for PDF book parsing. You can enhance it by:
1. Adding more sophisticated pattern recognition
2. Implementing page number detection
3. Adding error handling for malformed PDFs
4. Incorporating background processing with Celery
5. Adding content cleanup/normalization steps