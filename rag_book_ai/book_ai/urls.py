from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.signup_view, name='signup'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/delete-picture/', views.delete_profile_picture, name='delete_profile_picture'),
    path('upload/', views.upload_book, name='upload_book'),
    path('book/<int:book_id>/', views.view_book, name='view_book'),
    path('book/<int:book_id>/ask/', views.ask_question, name='ask_question'),
    path('book/<int:book_id>/clear-chat/', views.clear_chat, name='clear_chat'),
    path('book/<int:book_id>/store-question/', views.store_question, name='store_question'),
    path('book/<int:book_id>/delete/', views.delete_book, name='delete_book'),
    path('book/<int:book_id>/rename/', views.rename_book, name='rename_book'),
    path('book/<int:book_id>/download/', views.download_book, name='download_book'),
    path('chapter/<int:chapter_id>/frequent-questions/', views.get_frequent_questions, name='get_frequent_questions'),
    path('update-api-settings/', views.update_api_settings, name='update_api_settings'),
    path('rate-response/', views.rate_response, name='rate_response'),
    path('submit-feedback/', views.submit_feedback, name='submit_feedback'),
    
    # Enhanced API endpoints
    path('api/book-info/<int:book_id>/', views.api_book_info, name='api_book_info'),
    path('api/book-settings/<int:book_id>/', views.api_book_settings, name='api_book_settings'),
    path('api/book-update/<int:book_id>/', views.api_book_update, name='api_book_update'),
    path('api/book-delete/<int:book_id>/', views.api_book_delete, name='api_book_delete'),
    path('api/book-export/<int:book_id>/', views.api_book_export, name='api_book_export'),
    path('api/book-reprocess/<int:book_id>/', views.api_book_reprocess, name='api_book_reprocess'),
]
