from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=255)
    file_path = models.CharField(max_length=512)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    page_count = models.IntegerField(default=0)

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
