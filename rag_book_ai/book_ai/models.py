from django.db import models
from django.contrib.auth.models import User

class Book(models.Model):
    title = models.CharField(max_length=255)
    file_path = models.CharField(max_length=512)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    page_count = models.IntegerField(default=0)
    groq_api_key = models.CharField(max_length=255, blank=True)
    preferred_model = models.CharField(max_length=100, default="llama-3.3-70b-versatile")

    def __str__(self):
        return self.title

class BookChapter(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='chapters')
    title = models.CharField(max_length=255)
    level = models.IntegerField()
    start_page = models.IntegerField()
    end_page = models.IntegerField()
    content = models.TextField()
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='children')

    def __str__(self):
        return f"{self.book.title} - {self.title}"

class Question(models.Model):
    chapter = models.ForeignKey(BookChapter, on_delete=models.CASCADE, related_name='questions')
    text = models.TextField()
    frequency = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.chapter.title}: {self.text[:50]}..."
        
class ResponseRating(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='ratings')
    message_id = models.CharField(max_length=100)  # Client-generated ID for the message
    rating = models.CharField(max_length=20)  # 'useful' or 'not-useful'
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.book.title}: {self.rating} ({self.message_id})"
        
class ResponseFeedback(models.Model):
    rating = models.ForeignKey(ResponseRating, on_delete=models.CASCADE, related_name='feedback', null=True)
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='feedback')
    message_id = models.CharField(max_length=100)
    feedback_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.book.title}: Feedback for {self.message_id}"

from django.contrib.auth.models import User

# User Profile model for extended user information
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pics', blank=True, null=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pics', blank=True, null=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
