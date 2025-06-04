import os
import re
import tempfile
from typing import List, Dict, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    import pandas as pd
    import chromadb
    from sentence_transformers import SentenceTransformer
    from typing_extensions import Annotated, TypedDict
    
    # RAG and AI imports
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langgraph.graph import START, StateGraph
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph.message import add_messages
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False
        logger.warning("RAG features require additional packages. Install with: pip install langchain-groq langgraph sentence-transformers chromadb")
    
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    logger.warning("PDF processing libraries not available. Install with: pip install PyMuPDF pandas chromadb sentence-transformers")

# Supported Groq models for book teaching
SUPPORTED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant", 
    "gemma2-9b-it",
    "mixtral-8x7b-32768"
]

class BookTeachingRAG:
    def __init__(self):
        if not RAG_AVAILABLE:
            raise ImportError("RAG features not available")
            
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.groq_model = None
        self.app = None

    def setup_groq_model(self, api_key, model_name="llama-3.3-70b-versatile"):
        self.groq_model = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.7
        )
        self._initialize_workflow()

    def _initialize_workflow(self):
        # Define the state structure
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def chatbot(state: State):
            return {"messages": [self.groq_model.invoke(state["messages"])]}

        # Build the workflow
        workflow = StateGraph(State)
        workflow.add_node("chatbot", chatbot)
        workflow.add_edge(START, "chatbot")
        
        # Initialize memory
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

    def index_book_content(self, book_chunks):
        """Index book content for RAG retrieval"""
        try:
            # Create or get collection
            collection_name = f"book_content_{hash(str(book_chunks))}"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except:
                self.collection = self.chroma_client.create_collection(collection_name)

            # Create RAG chunks
            rag_chunks = self.create_rag_chunks(book_chunks)
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(rag_chunks):
                documents.append(chunk['content'])
                metadatas.append({
                    'title': chunk['title'],
                    'start_page': chunk['start_page'],
                    'end_page': chunk['end_page'],
                    'level': chunk['level']
                })
                ids.append(f"chunk_{i}")
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Indexed {len(rag_chunks)} chunks for RAG")
            
        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            raise

    def create_rag_chunks(self, book_chunks):
        """Create smaller chunks suitable for RAG"""
        rag_chunks = []
        
        for chunk in book_chunks:
            content = chunk['content']
            title = chunk['title']
            
            # Split long content into smaller chunks (max 1000 chars)
            if len(content) > 1000:
                words = content.split()
                current_chunk = []
                current_length = 0
                chunk_num = 1
                
                for word in words:
                    if current_length + len(word) > 1000 and current_chunk:
                        rag_chunks.append({
                            'title': f"{title} (Part {chunk_num})",
                            'content': ' '.join(current_chunk),
                            'start_page': chunk['start_page'],
                            'end_page': chunk['end_page'],
                            'level': chunk['level']
                        })
                        current_chunk = [word]
                        current_length = len(word)
                        chunk_num += 1
                    else:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                
                # Add remaining content
                if current_chunk:
                    rag_chunks.append({
                        'title': f"{title} (Part {chunk_num})" if chunk_num > 1 else title,
                        'content': ' '.join(current_chunk),
                        'start_page': chunk['start_page'],
                        'end_page': chunk['end_page'],
                        'level': chunk['level']
                    })
            else:
                rag_chunks.append(chunk)
        
        return rag_chunks

    def retrieve_context(self, query, chapter_filter=None):
        """Retrieve relevant context for a query"""
        if not self.collection:
            return []

        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            context = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                
                # Apply chapter filter if specified
                if chapter_filter and chapter_filter.lower() not in metadata['title'].lower():
                    continue
                
                context.append({
                    'content': doc,
                    'title': metadata['title'],
                    'start_page': metadata['start_page'],
                    'end_page': metadata['end_page']
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def teach_topic(self, user_question, messages_history, selected_chapter=None, thread_id="default"):
        """Generate teaching response using RAG"""
        try:
            # Retrieve relevant context
            context = self.retrieve_context(user_question, selected_chapter)
            
            # Build context string
            context_str = ""
            if context:
                context_str = "\n\nRelevant book content:\n"
                for ctx in context:
                    context_str += f"\n[{ctx['title']} - Pages {ctx['start_page']+1}-{ctx['end_page']+1}]\n{ctx['content']}\n"
            
            # Create system prompt
            system_prompt = f"""You are an AI teacher helping students understand a book. 
            Use the provided book content to answer questions accurately and pedagogically.
            
            Guidelines:
            - Base your answers on the provided book content
            - Explain concepts clearly and provide examples when helpful
            - If the question cannot be answered from the book content, say so
            - Always cite which chapter/section your answer comes from
            - Be encouraging and supportive in your teaching style
            
            {context_str}
            """
            
            # Prepare messages
            messages = [HumanMessage(content=system_prompt)]
            
            # Add conversation history
            for msg in messages_history[-5:]:  # Last 5 messages for context
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current question
            messages.append(HumanMessage(content=user_question))
            
            # Generate response using LangGraph
            config = {"configurable": {"thread_id": thread_id}}
            result = self.app.invoke({"messages": messages}, config)
            
            return result["messages"][-1].content
            
        except Exception as e:
            logger.error(f"Error generating teaching response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"


class PDFProcessor:
    def __init__(self, file_path):
        if not PDF_PROCESSING_AVAILABLE:
            raise ImportError("PDF processing libraries not available")
            
        self.file_path = file_path
        self.doc = None
        self.page_char_counts = []

    def open_document(self) -> bool:
        """Open the PDF document"""
        try:
            self.doc = fitz.open(self.file_path)
            return True
        except Exception as e:
            logger.error(f"Error opening document: {e}")
            return False

    def close_document(self) -> None:
        """Close the PDF document"""
        if self.doc:
            self.doc.close()
            self.doc = None

    def extract_text_fitz(self) -> str:
        """Extract all text from the PDF and track character counts per page"""
        if not self.doc:
            return ""
        
        full_text = ""
        self.page_char_counts = []
        
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            page_text = page.get_text()
            
            self.page_char_counts.append(len(full_text))
            full_text += page_text + "\n"
        
        # Add final count
        self.page_char_counts.append(len(full_text))
        return full_text

    def extract_bookmarks_fitz(self) -> List[Tuple[str, int, int]]:
        """Extract bookmarks from PDF"""
        if not self.doc:
            return []
        
        bookmarks = []
        toc = self.doc.get_toc()
        
        for i, item in enumerate(toc):
            level, title, page_num = item
            title = title.strip()
            page_num = max(0, page_num - 1)  # Convert to 0-indexed
            
            # Determine end page
            end_page = self.doc.page_count - 1
            for j in range(i + 1, len(toc)):
                if toc[j][0] <= level:  # Same or higher level
                    end_page = max(0, toc[j][2] - 2)  # Convert to 0-indexed
                    break
            
            bookmarks.append((title, page_num, end_page))
        
        return bookmarks

    def identify_chapters_regex(self, text: str) -> List[Tuple[str, int, int]]:
        """Identify chapters using regex patterns"""
        patterns = [
            r'^(Chapter\s+\d+[:\.]?\s*.*?)(?=\n|\r|$)',
            r'^(Part\s+[IVX]+[:\.]?\s*.*?)(?=\n|\r|$)',
            r'^(Section\s+\d+[:\.]?\s*.*?)(?=\n|\r|$)',
            r'^(\d+\.\s+[A-Z][^.]*?)(?=\n|\r|$)',
            r'^([A-Z][A-Z\s]{10,}?)(?=\n|\r|$)',
        ]
        
        chapter_starts = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                title = match.group(1).strip()
                char_index = match.start()
                page_num = self.find_page_number(char_index)
                
                # Avoid duplicates
                if not any(abs(existing[1] - page_num) < 2 for existing in chapter_starts):
                    chapter_starts.append((title, char_index, page_num))
        
        # Sort by character index and determine end pages
        chapter_starts.sort(key=lambda x: x[1])
        processed_chapters = []
        
        for i, (title, char_index, start_page) in enumerate(chapter_starts):
            if i + 1 < len(chapter_starts):
                end_page = chapter_starts[i + 1][2] - 1
            else:
                end_page = self.doc.page_count - 1
            
            processed_chapters.append((title, start_page, end_page))
        
        return processed_chapters

    def find_page_number(self, char_index: int) -> int:
        """Find page number for a given character index"""
        for i, count in enumerate(self.page_char_counts[:-1]):
            if char_index < self.page_char_counts[i + 1]:
                return i
        return len(self.page_char_counts) - 2

    def process_bookmark_chunks(self, bookmarks: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
        """Process bookmarks into content chunks"""
        chunks = []
        
        for i, (title, start_page, end_page) in enumerate(bookmarks):
            # Extract content for this chapter
            content = ""
            for page_num in range(start_page, min(end_page + 1, self.doc.page_count)):
                page = self.doc[page_num]
                content += page.get_text() + "\n"
            
            chunks.append({
                'id': f'bookmark_{i}',
                'title': title,
                'level': 1,  # Will be updated in hierarchical processing
                'start_page': start_page,
                'end_page': end_page,
                'content': content.strip(),
                'parent_id': None,
                'children': []
            })
        
        return chunks

    def process_regex_chunks(self, chapter_starts: List[Tuple[str, int, int]], full_text: str) -> List[Dict[str, Any]]:
        """Process regex-detected chapters into content chunks"""
        chunks = []
        
        for i, (title, start_page, end_page) in enumerate(chapter_starts):
            # Extract content for this chapter
            content = ""
            for page_num in range(start_page, min(end_page + 1, self.doc.page_count)):
                page = self.doc[page_num]
                content += page.get_text() + "\n"
            
            chunks.append({
                'id': f'regex_{i}',
                'title': title,
                'level': 1,
                'start_page': start_page,
                'end_page': end_page,
                'content': content.strip(),
                'parent_id': None,
                'children': []
            })
        
        return chunks

    def build_hierarchical_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build hierarchical relationships between chunks"""
        # For now, return chunks as-is
        # In a more sophisticated implementation, we would analyze titles and indentation
        return chunks

    def process_pdf(self) -> Optional[Dict[str, Any]]:
        """Main processing function"""
        try:
            if not self.open_document():
                return None
            
            # Extract full text
            full_text = self.extract_text_fitz()
            if not full_text.strip():
                logger.warning("No text extracted from PDF")
                return None
            
            # Try to extract bookmarks first
            bookmarks = self.extract_bookmarks_fitz()
            
            if bookmarks:
                logger.info(f"Found {len(bookmarks)} bookmarks")
                chunks = self.process_bookmark_chunks(bookmarks)
            else:
                logger.info("No bookmarks found, using regex detection")
                chapter_starts = self.identify_chapters_regex(full_text)
                chunks = self.process_regex_chunks(chapter_starts, full_text)
            
            # Build hierarchical structure
            hierarchical_chunks = self.build_hierarchical_chunks(chunks)
            
            result = {
                'filename': os.path.basename(self.file_path),
                'page_count': self.doc.page_count,
                'chunks': chunks,
                'hierarchical_chunks': hierarchical_chunks,
                'bookmarks': bookmarks,
                'full_text': full_text
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return None
        finally:
            self.close_document()
