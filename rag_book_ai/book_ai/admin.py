from django.contrib import admin
from .models import Book, BookChapter, Question, ResponseRating, ResponseFeedback, UserProfile

# Register your models here.
admin.site.register(Book)
admin.site.register(BookChapter)
admin.site.register(Question)
admin.site.register(ResponseRating)
admin.site.register(ResponseFeedback)
admin.site.register(UserProfile)
