#!/usr/bin/env python3
"""
Test script to verify the Book AI application improvements
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

# Add the Django project to Python path
django_path = Path(__file__).parent / "rag_book_ai"
sys.path.insert(0, str(django_path))

def test_django_server():
    """Test if Django server is running and responding"""
    try:
        response = requests.get('http://127.0.0.1:8000/', timeout=5)
        if response.status_code == 200:
            print("✓ Django server is running and responding")
            return True
        else:
            print(f"✗ Django server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Could not connect to Django server: {e}")
        return False

def test_database_connection():
    """Test database connection and models"""
    try:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_book_ai.settings')
        import django
        django.setup()
        
        from book_ai.models import Book, BookChapter, Question
        
        # Test basic model operations
        book_count = Book.objects.count()
        chapter_count = BookChapter.objects.count()
        question_count = Question.objects.count()
        
        print(f"✓ Database connection working")
        print(f"  - Books: {book_count}")
        print(f"  - Chapters: {chapter_count}")
        print(f"  - Questions: {question_count}")
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_rag_implementation():
    """Test the improved RAG implementation"""
    try:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_book_ai.settings')
        import django
        django.setup()
        
        from book_ai.utils.improved_book_teaching_rag import BookTeachingRAG
        
        # Test basic RAG initialization
        rag = BookTeachingRAG()
        print("✓ Improved RAG implementation loads successfully")
        
        # Test chunking functionality with sample data
        sample_chunks = [{
            'title': 'Test Chapter',
            'content': 'This is a test paragraph. ' * 100,  # Create content > 300 words
            'start_page': 1,
            'end_page': 5,
            'level': 1
        }]
        
        rag_chunks = rag.create_rag_chunks(sample_chunks)
        if len(rag_chunks) > 0:
            print(f"✓ Chunking functionality working - created {len(rag_chunks)} chunks")
            return True
        else:
            print("✗ Chunking functionality failed - no chunks created")
            return False
            
    except Exception as e:
        print(f"✗ RAG implementation test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    endpoints_to_test = [
        ('/', 'Home page'),
        ('/update-api-settings/', 'API settings endpoint (should allow POST)')
    ]
    
    results = []
    for endpoint, description in endpoints_to_test:
        try:
            if endpoint == '/update-api-settings/':
                # Test POST endpoint
                response = requests.post(f'http://127.0.0.1:8000{endpoint}', 
                                       json={'groq_api_key': 'test', 'model_name': 'test'},
                                       timeout=5)
                # We expect either 200 (success) or 403 (CSRF) for Django
                if response.status_code in [200, 403]:
                    print(f"✓ {description} - endpoint accessible")
                    results.append(True)
                else:
                    print(f"✗ {description} - unexpected status: {response.status_code}")
                    results.append(False)
            else:
                response = requests.get(f'http://127.0.0.1:8000{endpoint}', timeout=5)
                if response.status_code == 200:
                    print(f"✓ {description} - accessible")
                    results.append(True)
                else:
                    print(f"✗ {description} - status: {response.status_code}")
                    results.append(False)
        except Exception as e:
            print(f"✗ {description} - error: {e}")
            results.append(False)
    
    return all(results)

def main():
    """Run all tests"""
    print("🧪 Testing Book AI Application Improvements")
    print("=" * 50)
    
    tests = [
        ("Django Server", test_django_server),
        ("Database Connection", test_database_connection),
        ("RAG Implementation", test_rag_implementation),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Book AI application is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
