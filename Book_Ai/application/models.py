from django.db import models
from django.contrib.auth.models import User
import json

class Book(models.Model):
    title = models.CharField(max_length=255)
    filename = models.CharField(max_length=255)
    file_path = models.FileField(upload_to='books/')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    page_count = models.IntegerField(default=0)
    processing_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed')
        ],
        default='pending'
    )
    
    def __str__(self):
        return self.title

class Chapter(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='chapters')
    title = models.CharField(max_length=500)
    level = models.IntegerField(default=1)  # 1=chapter, 2=section, 3=subsection, etc.
    start_page = models.IntegerField()
    end_page = models.IntegerField()
    content = models.TextField()
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='children')
    chapter_id = models.CharField(max_length=50, null=True, blank=True)
    
    class Meta:
        ordering = ['start_page', 'level']
    
    def __str__(self):
        return f"{self.book.title} - {self.title}"

class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Chat Session {self.session_id} for {self.book.title}"

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(
        max_length=10,
        choices=[
            ('human', 'Human'),
            ('ai', 'AI')
        ]
    )
    content = models.TextField()
    context_chapters = models.ManyToManyField(Chapter, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.message_type.upper()}: {self.content[:50]}..."
