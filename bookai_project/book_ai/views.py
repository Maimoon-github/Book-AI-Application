# book_ai/views.py

from django.shortcuts import render
from .rag_core import get_answer

def home(request):
    answer = None
    query = None
    if request.method == 'POST':
        query = request.POST.get('query')
        if query:
            answer = get_answer(query)
    return render(request, 'book_ai/home.html', {'answer': answer, 'query': query})