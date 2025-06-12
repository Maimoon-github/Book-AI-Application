from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.signup_view, name='signup'),
    path('profile/', views.profile_view, name='profile'),
    path('upload/', views.upload_book, name='upload_book'),
    path('book/<int:book_id>/', views.view_book, name='view_book'),
    path('book/<int:book_id>/ask/', views.ask_question, name='ask_question'),
    path('book/<int:book_id>/clear-chat/', views.clear_chat, name='clear_chat'),
    path('book/<int:book_id>/store-question/', views.store_question, name='store_question'),
    path('chapter/<int:chapter_id>/frequent-questions/', views.get_frequent_questions, name='get_frequent_questions'),
    path('update-api-settings/', views.update_api_settings, name='update_api_settings'),
    path('rate-response/', views.rate_response, name='rate_response'),
    path('submit-feedback/', views.submit_feedback, name='submit_feedback'),
]
