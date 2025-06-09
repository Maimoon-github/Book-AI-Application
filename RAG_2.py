import re
import math
import uuid
from collections import Counter
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Function to augment chunks with more detailed metadata
def augment_chunk_metadata(chunks):
    """
    Enriches each chunk with more detailed metadata for better retrieval.
    
    Args:
        chunks (list): List of chunk objects from hierarchical_chunking
        
    Returns:
        list: Same chunks with enriched metadata
    """
    # Make a copy of chunks to avoid modifying the input directly
    enriched_chunks = chunks.copy()
    
    # Download stopwords if not available
    try:
        stopwords.words('english')
    except:
        nltk.download('stopwords')
    
    try:
        sent_tokenize("Test sentence.")
    except:
        nltk.download('punkt')
    
    stop_words = set(stopwords.words('english'))
    
    # Section pattern for matching potential section headers
    section_pattern = r'(?:^|\n)(\d+\.\d+(?:\.\d+)*)\s+([^\n]+)'
    
    # Dictionary to store all words for TF-IDF calculation
    all_words = Counter()
    chunk_word_counts = []
    
    # First pass: gather word counts from all chunks
    for chunk in chunks:
        text = chunk["text_content"]
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) 
                if w.lower() not in stop_words]
        chunk_words = Counter(words)
        all_words.update(chunk_words)
        chunk_word_counts.append(chunk_words)
    
    # Calculate IDF
    num_chunks = len(chunks)
    word_idf = {}
    for word, count in all_words.items():
        # Count how many chunks contain this word
        chunks_with_word = sum(1 for chunk_words in chunk_word_counts if word in chunk_words)
        # IDF formula: log(total chunks / chunks containing word)
        word_idf[word] = math.log(num_chunks / (1 + chunks_with_word))
    
    # Second pass: augment metadata
    for i, chunk in enumerate(enriched_chunks):
        text = chunk["text_content"]
        chunk_id = str(uuid.uuid4())
        
        # Initialize enhanced metadata dict with existing metadata
        enhanced_metadata = chunk["metadata"].copy()
        
        # Add chunk ID
        enhanced_metadata["chunk_id"] = chunk_id
        
        # 1. Extract section information if available
        section_match = re.search(section_pattern, text)
        if section_match:
            enhanced_metadata["section_number"] = section_match.group(1)
            enhanced_metadata["section_title"] = section_match.group(2).strip()
        
        # 2. Extract keywords using TF-IDF
        chunk_words = chunk_word_counts[i]
        word_scores = {word: count * word_idf[word] for word, count in chunk_words.items()}
        keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        enhanced_metadata["keywords"] = [word for word, _ in keywords]
        
        # 3. Estimate difficulty level based on text complexity
        # Simple heuristic: average word length and sentence length
        avg_word_length = sum(len(word) for word in text.split()) / max(1, len(text.split()))
        sentences = sent_tokenize(text)
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / max(1, len(sentences))
        
        # Simple scoring
        if avg_word_length < 5 and avg_sentence_length < 10:
            difficulty = "beginner"
        elif avg_word_length < 6 and avg_sentence_length < 15:
            difficulty = "intermediate"
        else:
            difficulty = "advanced"
            
        enhanced_metadata["difficulty_level"] = difficulty
        
        # Add potential learning objective template
        chapter_title = enhanced_metadata.get("chapter_title", "")
        enhanced_metadata["learning_objective"] = f"Understand concepts related to {chapter_title}"
        
        # Update chunk with enhanced metadata
        chunk["metadata"] = enhanced_metadata

    return enriched_chunks
chunks = input("D:\Amin Data\SOFTWARE\SUGA Data\Dn\Maxon Cinema 4D R19 - WIN\Cinema 4D R19\R19_Install_Web_Win\Installation Guide R19 US.pdf")

# Augment chunks with detailed metadata
augmented_chunks = augment_chunk_metadata(chunks)

# Display sample of augmented chunks
print(f"Enhanced {len(augmented_chunks)} chunks with detailed metadata")
if augmented_chunks:
    print("\nSample chunk metadata:")
    print(json.dumps(augmented_chunks[0]["metadata"], indent=2))
    if len(augmented_chunks) > 1:
        print("\nSample chunk metadata:")
        print(json.dumps(augmented_chunks[1]["metadata"], indent=2))