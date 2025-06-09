from django.test import TestCase, Client, override_settings
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.messages import get_messages
from .models import Document, Chunk
from .rag_utils import HierarchicalChunker, MetadataEnricher, HybridRetriever
import json
import os
import tempfile
import numpy as np

# Create a temporary media root for testing
MEDIA_ROOT = tempfile.mkdtemp()

class RagModelTests(TestCase):
    def test_document_creation(self):
        """Test that a document can be created with the correct fields."""
        doc = Document.objects.create(
            name="Test Document",
            content="This is a test document content."
        )
        self.assertEqual(doc.name, "Test Document")
        self.assertEqual(doc.content, "This is a test document content.")
        
    def test_chunk_creation(self):
        """Test that a chunk can be created with the correct relationships."""
        doc = Document.objects.create(name="Test Document")
        metadata = {
            "chapter_number": 1, 
            "chapter_title": "Introduction",
            "keywords": ["test", "example", "sample"]
        }
        
        chunk = Chunk.objects.create(
            document=doc,
            text_content="This is a test chunk.",
            metadata=metadata
        )
        
        self.assertEqual(chunk.document, doc)
        self.assertEqual(chunk.text_content, "This is a test chunk.")
        self.assertEqual(chunk.metadata["chapter_number"], 1)
        self.assertEqual(chunk.metadata["keywords"][0], "test")

class RagViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        
    def test_home_view(self):
        """Test that the home view loads correctly."""
        response = self.client.get(reverse('book_ai:home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'book_ai/home.html')
          def test_upload_view(self):
        """Test that the upload view loads correctly."""
        response = self.client.get(reverse('book_ai:upload_document'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'book_ai/upload.html')

    @patch('book_ai.views.HierarchicalChunker')
    @patch('book_ai.views.MetadataEnricher')
    @patch('book_ai.views.initialize_retriever')
    @override_settings(MEDIA_ROOT=tempfile.mkdtemp())
    def test_document_upload_processing(self, mock_init_retriever, mock_enricher, mock_chunker):
        """Test document upload and processing flow with mocks."""
        # Setup mocks
        mock_chunks = [{"text_content": "Test chunk", "metadata": {}}]
        mock_chunker_instance = mock_chunker.return_value
        mock_chunker_instance.process_document.return_value = mock_chunks
        
        mock_enricher_instance = mock_enricher.return_value
        mock_enricher_instance.enrich_chunks.return_value = mock_chunks
        
        # Create a test PDF file
        test_file = SimpleUploadedFile(
            "test_document.pdf",
            b"Test PDF content",
            content_type="application/pdf"
        )
        
        # Upload the document
        response = self.client.post(
            reverse('book_ai:upload_document'),
            {'file': test_file},
            format='multipart'
        )
        
        # Check that mocks were called
        mock_chunker_instance.process_document.assert_called_once()
        mock_enricher_instance.enrich_chunks.assert_called_once_with(mock_chunks)
        mock_init_retriever.assert_called_once()
        
        # Check redirect
        self.assertRedirects(response, reverse('book_ai:home'))
        
        # Test document was created in DB
        self.assertEqual(Document.objects.count(), 1)
        self.assertEqual(Chunk.objects.count(), 1)

    def test_home_view_question_answering(self):
        """Test question answering functionality."""
        # Set up a document and chunks for testing
        doc = Document.objects.create(
            name="Test Document",
            content="This is test content."
        )
        
        Chunk.objects.create(
            document=doc,
            text_content="Python is a programming language.",
            metadata={"keywords": ["python", "programming"]}
        )
        
        # Initialize retriever
        initialize_retriever()
        
        # Submit a question
        response = self.client.post(
            reverse('book_ai:home'),
            {'query': 'What is Python?'}
        )
        
        # Check the response contains our question
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'What is Python?')
