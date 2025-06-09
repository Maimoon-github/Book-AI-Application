# Test script to verify Django server and RAG functionality
import os
import sys
import importlib
import requests
import json
from time import sleep

def print_status(message, status="INFO"):
    """Print a formatted status message"""
    status_colors = {
        "INFO": "\033[94m",  # Blue
        "OK": "\033[92m",    # Green
        "ERROR": "\033[91m", # Red
        "WARN": "\033[93m",  # Yellow
    }
    reset = "\033[0m"
    
    color = status_colors.get(status, "\033[0m")
    print(f"{color}[{status}]{reset} {message}")

def check_dependencies():
    """Check if all important dependencies are installed"""
    print_status("Checking dependencies...", "INFO")
    
    dependencies = [
        ("django", "Django framework"),
        ("numpy", "NumPy for numerical operations"),
        ("faiss", "FAISS for vector search"),
        ("sentence_transformers", "Sentence Transformers for embeddings"),
        ("nltk", "NLTK for text processing"),
        ("rank_bm25", "BM25 for keyword search"),
        ("PyPDF2", "PyPDF2 for PDF processing"),
        ("transformers", "HuggingFace Transformers for text generation"),
    ]
    
    all_ok = True
    for module_name, description in dependencies:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                print_status(f"{description} ({module_name}): Not installed", "WARN")
                all_ok = False
            else:
                try:
                    module = importlib.import_module(module_name)
                    version = getattr(module, "__version__", "unknown")
                    print_status(f"{description} ({module_name}): Installed (version: {version})", "OK")
                except Exception as e:
                    print_status(f"{description} ({module_name}): Error checking version: {e}", "WARN")
        except Exception as e:
            print_status(f"Error checking {module_name}: {e}", "ERROR")
            all_ok = False
    
    return all_ok

def test_django_server():
    """Test if the Django server is running"""
    print_status("\nTesting Django server...", "INFO")
    
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print_status("Django server is running", "OK")
            return True
        else:
            print_status(f"Django server returned status code: {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status("Could not connect to Django server", "ERROR")
        print_status("Please start the server with: python manage.py runserver", "INFO")
        return False

def test_rag_functionality():
    """Test basic RAG functionality"""
    print_status("\nTesting RAG functionality...", "INFO")
    
    # Try to import the RAG core module directly for a quick test
    try:
        sys.path.append(os.path.join(os.getcwd(), "book_ai"))
        from book_ai import rag_core
        
        retriever_type = getattr(rag_core, "retriever_type", "unknown")
        print_status(f"Retriever type: {retriever_type}", "INFO")
        
        if retriever_type == "faiss":
            print_status("Using full-featured FAISS retriever", "OK")
        elif retriever_type == "simple":
            print_status("Using SimpleHybridRetriever with BM25", "INFO")
        elif retriever_type == "basic":
            print_status("Using minimal BasicRetriever (limited features)", "WARN")
        else:
            print_status(f"Unknown retriever type: {retriever_type}", "WARN")
        
        return True
    except Exception as e:
        print_status(f"Error testing RAG functionality: {e}", "ERROR")
        return False

def test_file_upload():
    """Test file upload API if server is running"""
    print_status("\nTesting file upload API...", "INFO")
    
    try:
        # First check if server is running
        response = requests.get("http://localhost:8000/upload/")
        if response.status_code != 200:
            print_status("Upload page not available", "WARN")
            return False
        
        print_status("Upload page is available", "OK")
        return True
    except requests.exceptions.ConnectionError:
        print_status("Could not connect to server", "ERROR")
        return False

if __name__ == "__main__":
    print_status("Starting system test", "INFO")
    deps_ok = check_dependencies()
    server_ok = test_django_server()
    
    if server_ok:
        rag_ok = test_rag_functionality()
        upload_ok = test_file_upload()
    else:
        rag_ok = False
        upload_ok = False
    
    print("\n=== Test Results ===")
    print_status(f"Dependencies check: {'PASSED' if deps_ok else 'WARNINGS REPORTED'}", "OK" if deps_ok else "WARN")
    print_status(f"Django server: {'RUNNING' if server_ok else 'NOT RUNNING'}", "OK" if server_ok else "ERROR")
    if server_ok:
        print_status(f"RAG functionality: {'OK' if rag_ok else 'ISSUES FOUND'}", "OK" if rag_ok else "WARN")
        print_status(f"File upload API: {'AVAILABLE' if upload_ok else 'NOT AVAILABLE'}", "OK" if upload_ok else "WARN")
    
    if not server_ok:
        print("\nTo start the server, run:")
        print("  python manage.py runserver")
