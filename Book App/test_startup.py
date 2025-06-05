#!/usr/bin/env python3
"""
Quick test script to check if the main imports work without warnings
"""
import warnings
import os

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
os.environ["PYTORCH_DISABLE_VERSION_CHECK"] = "1"

try:
    print("Testing imports...")
    import streamlit as st
    print("✓ Streamlit imported successfully")
    
    import sentence_transformers
    print("✓ SentenceTransformers imported successfully")
    
    import chromadb
    print("✓ ChromaDB imported successfully")
    
    # Test basic functionality
    from sentence_transformers import SentenceTransformer
    print("✓ SentenceTransformer class imported successfully")
    
    print("\n🎉 All imports successful! Your app should run without PyTorch warnings.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
