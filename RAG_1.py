import re
try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    import subprocess
    subprocess.check_call(["python", '-m', 'pip', 'install', 'nltk'])
    import nltk
    from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch

# Check if PyPDF2 is installed, if not install it
try:
    import PyPDF2
except ImportError:
    import subprocess
    subprocess.check_call(["python", '-m', 'pip', 'install', 'PyPDF2'])
    import PyPDF2

# Download necessary NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
    # Also check for punkt_tab specifically
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    # Download punkt_tab as well, as indicated by the error
    nltk.download('punkt_tab')


def hierarchical_chunking(content, semantic_threshold=0.6, overlap_sentences=1, max_chunk_sentences=15):
    """
    Performs hierarchical chunking on educational content:
    1. Splits content into chapters
    2. Further splits chapters into semantic chunks

    Args:
        content (str): Raw educational text content
        semantic_threshold (float): Threshold for semantic similarity between sentences
        overlap_sentences (int): Number of sentences to overlap between chunks
        max_chunk_sentences (int): Maximum sentences per chunk

    Returns:
        list: List of chunk objects with text content and metadata
    """
    # Load sentence transformer model for semantic analysis
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Step 1: Extract chapters
    chapters_data = extract_chapters(content)

    # Step 2: Process each chapter into semantic chunks
    all_chunks = []
    for chapter in chapters_data:
        chapter_chunks = semantic_chunk_chapter(
            chapter["text"],
            chapter["number"],
            chapter["title"],
            model,
            semantic_threshold,
            overlap_sentences,
            max_chunk_sentences
        )
        all_chunks.extend(chapter_chunks)

    return all_chunks

def extract_chapters(content):
    """Extracts individual chapters from the content."""
    # Match patterns like "Chapter X:", "Chapter X -", etc.
    chapter_pattern = r'(?:Chapter|CHAPTER)\s+(\d+)[\s:-]+\s*([^\n]+)'

    # Find all chapter beginnings
    chapter_matches = list(re.finditer(chapter_pattern, content))

    chapters_data = []

    # Process each chapter
    for i, match in enumerate(chapter_matches):
        chapter_number = int(match.group(1))
        chapter_title = match.group(2).strip()
        start_pos = match.start()

        # Determine end position (either the next chapter start or end of content)
        end_pos = chapter_matches[i+1].start() if i < len(chapter_matches) - 1 else len(content)

        # Extract the full chapter text (including its title)
        chapter_text = content[start_pos:end_pos].strip()

        chapters_data.append({
            "number": chapter_number,
            "title": chapter_title,
            "text": chapter_text
        })

    # If no chapters were found, treat the entire content as one chapter
    if not chapters_data:
        chapters_data.append({
            "number": 1,
            "title": "Content",
            "text": content
        })

    return chapters_data

def semantic_chunk_chapter(chapter_text, chapter_number, chapter_title, model,
                           semantic_threshold=0.6, overlap_sentences=1, max_chunk_sentences=15):
    """
    Divides a chapter into semantically coherent chunks.

    Args:
        chapter_text (str): The full text of the chapter
        chapter_number (int): The chapter number
        chapter_title (str): The chapter title
        model: The sentence transformer model
        semantic_threshold (float): Similarity threshold to determine chunk boundaries
        overlap_sentences (int): Number of sentences to overlap between chunks
        max_chunk_sentences (int): Maximum number of sentences per chunk

    Returns:
        list: List of chunk objects for this chapter
    """
    # Tokenize the chapter into sentences
    sentences = sent_tokenize(chapter_text)
    if len(sentences) == 0:
        return []

    # Get embeddings for all sentences
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk_sentences = []
    current_sentence_count = 0

    for i in range(len(sentences)):
        current_chunk_sentences.append(sentences[i])
        current_sentence_count += 1

        # Create a new chunk if:
        # 1. We've reached the maximum chunk size, or
        # 2. We detect a semantic boundary and have a minimum chunk size
        create_new_chunk = current_sentence_count >= max_chunk_sentences

        if i < len(sentences) - 1 and not create_new_chunk and current_sentence_count >= 3:
            # Calculate similarity between current sentence and next
            similarity = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i+1]).item()
            # If similarity is below threshold, it's a good chunk boundary
            if similarity < semantic_threshold:
                create_new_chunk = True

        if create_new_chunk or i == len(sentences) - 1:
            # Add the current chunk
            chunks.append({
                "text_content": " ".join(current_chunk_sentences),
                "metadata": {
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "chunk_index": len(chunks)
                }
            })

            # Start a new chunk, possibly with overlap
            if i < len(sentences) - 1:  # Only if not the last sentence
                overlap_start = max(0, len(current_chunk_sentences) - overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                current_sentence_count = len(current_chunk_sentences)

    return chunks

# Example usage:
pdf_path = input(r"")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

my_textbook_content = extract_text_from_pdf(pdf_path)
print(f"Extracted {len(my_textbook_content)} characters from PDF")

# Process the content into chunks
chunks = hierarchical_chunking(my_textbook_content)
print(f"Created {len(chunks)} chunks from the textbook")