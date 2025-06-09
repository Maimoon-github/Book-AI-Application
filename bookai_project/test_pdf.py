import sys
from book_ai.rag_utils import HierarchicalChunker

def test_pdf_extraction(pdf_path):
    chunker = HierarchicalChunker()
    with open(pdf_path, 'rb') as f:
        try:
            text = chunker.extract_text_from_pdf(f)
            print("Successfully extracted text from PDF!")
            print("\nFirst 500 characters:", text[:500])
            print(f"\nTotal extracted text length: {len(text)} characters")
        except Exception as e:
            print(f"Error extracting text: {e}", file=sys.stderr)

if __name__ == '__main__':
    test_pdf_extraction('media/documents/Introductory_Statistics.pdf')
