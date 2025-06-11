from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_book, name='upload_book'),
    path('book/<int:book_id>/', views.view_book, name='view_book'),
    path('book/<int:book_id>/ask/', views.ask_question, name='ask_question'),
    path('book/<int:book_id>/clear-chat/', views.clear_chat, name='clear_chat'),
    path('chapter/<int:chapter_id>/frequent-questions/', views.get_frequent_questions, name='get_frequent_questions'),
]
