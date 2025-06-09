from django.urls import path
from . import views

app_name = 'book_ai'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_document, name='upload_document'),
]
