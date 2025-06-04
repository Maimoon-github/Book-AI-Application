from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_book, name='upload_book'),
    path('book/<int:book_id>/', views.book_detail, name='book_detail'),
    path('book/<int:book_id>/chat/', views.chat_interface, name='chat_interface'),
    path('book/<int:book_id>/search/', views.search_content, name='search_content'),
    path('book/<int:book_id>/export/', views.export_structure, name='export_structure'),
    path('book/<int:book_id>/progress/', views.update_reading_progress, name='update_reading_progress'),
    path('book/<int:book_id>/update/', views.update_book_info, name='update_book_info'),
    path('chapter/<int:chapter_id>/content/', views.chapter_content, name='chapter_content'),
    path('api/chat/<int:book_id>/send/', views.send_message, name='send_message'),
    path('api/recommendations/', views.book_recommendation, name='book_recommendation'),
]
