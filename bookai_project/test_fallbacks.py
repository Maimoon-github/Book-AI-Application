# Test the dependency fallback mechanism
import sys
import unittest
import importlib
from unittest.mock import patch

class DependencyFallbackTests(unittest.TestCase):
    """Tests for the dependency fallback mechanisms in our RAG implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear modules that might have been imported
        modules_to_remove = ['book_ai.rag_utils', 'book_ai.basic_retriever', 'book_ai.simple_retriever', 'book_ai.rag_core']
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]
    
    def test_import_basic_retriever(self):
        """Test that basic_retriever imports without dependencies."""
        try:
            from book_ai.basic_retriever import BasicRetriever
            # Create a retriever instance
            retriever = BasicRetriever()
            self.assertIsNotNone(retriever)
            print("✅ BasicRetriever imports successfully without dependencies")
        except Exception as e:
            self.fail(f"BasicRetriever failed to import: {e}")

    def test_dependencies_status(self):
        """Test that DEPENDENCIES_STATUS is properly tracked in rag_utils."""
        try:
            from book_ai.rag_utils import DEPENDENCIES_STATUS
            self.assertIsNotNone(DEPENDENCIES_STATUS)
            self.assertIsInstance(DEPENDENCIES_STATUS, dict)
            print(f"✅ DEPENDENCIES_STATUS properly tracked with {len(DEPENDENCIES_STATUS)} items")
            
            # Print the current status
            for dep, status in DEPENDENCIES_STATUS.items():
                status_str = "Available" if status else "Missing"
                print(f"  - {dep}: {status_str}")
        except Exception as e:
            self.fail(f"Failed to check DEPENDENCIES_STATUS: {e}")
    
    def test_simple_hybrid_retriever_fallbacks(self):
        """Test SimpleHybridRetriever's fallback mechanism."""
        try:
            from book_ai.simple_retriever import SimpleHybridRetriever
            # Create a retriever instance
            retriever = SimpleHybridRetriever()
            self.assertIsNotNone(retriever)
            print("✅ SimpleHybridRetriever imports successfully")
            
            # Test the add_chunks method with empty input
            retriever.add_chunks([])
            
            # Test search with no chunks
            results = retriever.search("test query")
            self.assertEqual(len(results), 0)
            print("✅ SimpleHybridRetriever handles empty searches gracefully")
        except Exception as e:
            self.fail(f"SimpleHybridRetriever test failed: {e}")
    
    def test_rag_core_retriever_selection(self):
        """Test retriever selection in rag_core."""
        try:
            from book_ai.rag_core import retriever_type, RetrieverClass
            self.assertIsNotNone(retriever_type)
            self.assertIsNotNone(RetrieverClass)
            print(f"✅ rag_core selected retriever: {retriever_type} ({RetrieverClass.__name__})")
        except Exception as e:
            self.fail(f"rag_core retriever selection test failed: {e}")

if __name__ == "__main__":
    # Set up Django environment
    import os
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bookai_project.settings")
    django.setup()
    
    print("Running dependency fallback tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nTests complete.")
