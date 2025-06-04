from django.core.management.base import BaseCommand
from application.models import Book, Chapter
from application.utils import PDFProcessor, PDF_PROCESSING_AVAILABLE
import os

class Command(BaseCommand):
    help = 'Test PDF processing functionality'

    def add_arguments(self, parser):
        parser.add_argument('pdf_path', type=str, help='Path to PDF file to test')

    def handle(self, *args, **options):
        pdf_path = options['pdf_path']
        
        if not PDF_PROCESSING_AVAILABLE:
            self.stdout.write(
                self.style.ERROR('PDF processing not available. Install required packages.')
            )
            return
        
        if not os.path.exists(pdf_path):
            self.stdout.write(
                self.style.ERROR(f'File not found: {pdf_path}')
            )
            return
        
        self.stdout.write(f'Processing PDF: {pdf_path}')
        
        processor = PDFProcessor(pdf_path)
        result = processor.process_pdf()
        
        if not result:
            self.stdout.write(
                self.style.ERROR('Failed to process PDF')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully processed: {result["filename"]}')
        )
        self.stdout.write(f'Pages: {result["page_count"]}')
        self.stdout.write(f'Chapters found: {len(result["hierarchical_chunks"])}')
        
        for i, chunk in enumerate(result["hierarchical_chunks"][:5]):  # Show first 5
            self.stdout.write(f'{i+1}. {chunk["title"]} (Pages {chunk["start_page"]+1}-{chunk["end_page"]+1})')
        
        if len(result["hierarchical_chunks"]) > 5:
            self.stdout.write(f'... and {len(result["hierarchical_chunks"]) - 5} more chapters')
