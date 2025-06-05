import streamlit as st
import os
import re
import tempfile
from typing import List, Dict, Tuple, Optional, Any, Sequence
import pandas as pd
import fitz  # PyMuPDF

# RAG and AI imports
import chromadb
from sentence_transformers import SentenceTransformer
from typing_extensions import Annotated, TypedDict

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
    st.warning("RAG features require additional packages. Install with: pip install langchain-groq langgraph sentence-transformers chromadb")

# Supported Groq models for book teaching
SUPPORTED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant", 
    "gemma2-9b-it",
    "mixtral-8x7b-32768"
]

class BookTeachingRAG:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.groq_model = None
        self.app = None
        
    def setup_groq_model(self, api_key, model_name="llama-3.3-70b-versatile"):
        """Initialize Groq model for teaching responses"""
        self.groq_model = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.7
        )
        self._initialize_workflow()
    
    def _initialize_workflow(self):
        """Set up LangGraph workflow for teaching conversations"""
        teaching_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI teacher specializing in adaptive learning from books. Your teaching follows a structured pedagogical approach.

## Teaching Philosophy & Pattern:

### 🎯 **Learning Objective Identification**
- First understand what the student wants to learn
- Identify their current knowledge level
- Set clear, achievable learning goals

### 📚 **Structured Teaching Pattern**
When teaching any concept, follow this pattern:

1. **FOUNDATION** 🏗️
   - Start with core definitions and basic concepts
   - Provide clear, simple explanations
   - Use analogies from everyday life

2. **CONTEXT** 🌍
   - Explain where this fits in the bigger picture
   - Connect to previously learned material
   - Show relevance and importance

3. **EXAMPLES** 💡
   - Provide concrete, relatable examples
   - Use case studies from the book content
   - Show practical applications

4. **ANALYSIS** 🔍
   - Break down complex ideas into components
   - Explain cause-and-effect relationships
   - Highlight patterns and principles

5. **APPLICATION** 🚀
   - Suggest how to apply this knowledge
   - Provide practice scenarios
   - Connect to real-world situations

6. **REINFORCEMENT** 🎯
   - Summarize key takeaways
   - Suggest follow-up questions for deeper learning
   - Recommend related topics to explore

### 💬 **Communication Style**
- **Clarity**: Use simple, clear language
- **Engagement**: Ask thought-provoking questions
- **Encouragement**: Maintain positive, supportive tone
- **Adaptation**: Adjust complexity based on student responses
- **Source-Grounded**: Always reference the book content

### 🔄 **Interactive Learning**
- Ask "Do you understand?" or "What would you like to explore further?"
- Encourage questions and clarification requests
- Provide multiple perspectives on complex topics
- Use Socratic method when appropriate

### 📖 **Content Integration**
- Always ground explanations in the provided book context
- Reference specific chapters and page numbers
- Quote relevant passages when helpful
- Maintain fidelity to the author's intent

### 🎓 **Assessment & Progress**
- Check understanding with gentle questioning
- Provide positive reinforcement for engagement
- Suggest next steps in the learning journey
- Identify knowledge gaps and address them

Remember: You are not just answering questions - you are facilitating deep, meaningful learning experiences based on the book's content."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        class TeachingState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            context: str
            sources: list
        
        workflow = StateGraph(state_schema=TeachingState)
        
        def call_teaching_model(state: TeachingState):
            # Include context in the conversation
            context_message = f"\n--- BOOK CONTEXT ---\n{state.get('context', '')}\n--- END CONTEXT ---\n"
            
            # Prepare messages with context
            messages_with_context = list(state["messages"])
            if messages_with_context and isinstance(messages_with_context[-1], HumanMessage):
                # Add context to the last human message
                last_msg = messages_with_context[-1]
                enhanced_content = f"{context_message}\nSTUDENT QUESTION: {last_msg.content}"
                messages_with_context[-1] = HumanMessage(content=enhanced_content)
            
            prompt = teaching_prompt.invoke({"messages": messages_with_context})
            response = self.groq_model.invoke(prompt)
            return {"messages": [response]}
        
        workflow.add_node("teaching_model", call_teaching_model)
        workflow.add_edge(START, "teaching_model")
        
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)
    
    def index_book_content(self, book_chunks):
        """Store book chunks in vector database"""
        try:
            self.collection = self.chroma_client.create_collection(
                name="book_content", 
                get_or_create=True
            )
        except:
            self.collection = self.chroma_client.get_collection(name="book_content")
        
        rag_chunks = self.create_rag_chunks(book_chunks)
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(rag_chunks):
            embedding = self.embedding_model.encode(chunk['text'])
            
            documents.append(chunk['text'])
            embeddings.append(embedding.tolist())
            metadatas.append(chunk['metadata'])
            ids.append(f"chunk_{i}")
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def create_rag_chunks(self, book_chunks):
        """Split large chapters into smaller, contextual chunks using overlapping sliding windows"""
        rag_chunks = []

        window_size = 400  # max words per chunk
        overlap_size = 50  # words to overlap between chunks

        for chapter in book_chunks:
            words = chapter['content'].split()
            total_words = len(words)
            start_idx = 0

            while start_idx < total_words:
                end_idx = min(start_idx + window_size, total_words)
                chunk_words = words[start_idx:end_idx]
                chunk_text = ' '.join(chunk_words)

                rag_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chapter': chapter['title'],
                        'start_page': chapter['start_page'],
                        'end_page': chapter['end_page'],
                        'chunk_type': 'content'
                    }
                })

                # Stop if reached end
                if end_idx == total_words:
                    break

                # Move window by window_size - overlap_size
                start_idx += window_size - overlap_size

        return rag_chunks
    
    def retrieve_context(self, query, chapter_filter=None):
        """Retrieve most relevant chunks for the query"""
        if not self.collection:
            return {"documents": [[]], "metadatas": [[]]}
            
        query_embedding = self.embedding_model.encode(query)
        
        where_clause = None
        if chapter_filter:
            where_clause = {"chapter": {"$eq": chapter_filter}}
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            where=where_clause
        )
        
        return results
    
    def teach_topic(self, user_question, messages_history, selected_chapter=None, thread_id="default"):
        """Main teaching function using RAG and LangGraph"""
        # Retrieve relevant context
        context_data = self.retrieve_context(user_question, selected_chapter)
        
        # Prepare context text
        context_text = ""
        source_info = []
        
        for i, doc in enumerate(context_data['documents'][0]):
            if i < len(context_data['metadatas'][0]):
                metadata = context_data['metadatas'][0][i]
                context_text += f"\n--- Context {i+1} ---\n{doc}\n"
                source_info.append(f"Chapter: {metadata['chapter']}, Pages: {metadata['start_page']}-{metadata['end_page']}")
        
        # Prepare state for LangGraph
        state = {
            "messages": messages_history + [HumanMessage(content=user_question)],
            "context": context_text,
            "sources": source_info
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Generate response using LangGraph
        output = self.app.invoke(state, config)
        ai_response = output["messages"][-1]
        
        return {
            "response": ai_response,
            "sources": source_info,
            "context_used": len(context_data['documents'][0])
        }

class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.document = None
        self.filename = os.path.basename(file_path)
        self.file_extension = os.path.splitext(self.filename)[1].lower()
        self.page_char_counts = []  # Store character count per page for faster lookups

    def open_document(self) -> bool:
        """Opens the PDF document using PyMuPDF (fitz)."""
        try:
            if self.file_extension == ".pdf":
                self.document = fitz.open(self.file_path)
                return True
            else:
                raise ValueError(f"Unsupported file type: {self.file_extension}")
        except Exception as e:
            st.error(f"Error opening {self.filename}: {e}")
            return False

    def close_document(self) -> None:
        """Closes the PDF document."""
        if self.document:
            self.document.close()
            self.document = None

    def extract_text_fitz(self) -> str:
        """Extracts text from PDF using PyMuPDF (fitz)."""
        full_text = ""
        self.page_char_counts = []
        try:
            if not self.document:
                raise ValueError("Document not open.")
            for page in self.document:
                page_text = page.get_text()
                full_text += page_text
                self.page_char_counts.append(len(page_text))
            return full_text
        except Exception as e:
            st.error(f"Error extracting text from {self.filename}: {e}")
            return ""

    def extract_bookmarks_fitz(self) -> List[Tuple[str, int, int]]:
        """Extracts bookmarks using PyMuPDF (fitz)."""
        try:
            if not self.document:
                raise ValueError("Document not open.")
            
            toc = self.document.get_toc()
            if not toc:
                return []
                
            bookmarks = []
            for level, title, page in toc:
                # PyMuPDF uses 1-indexed page numbers, convert to 0-indexed
                page_idx = page - 1 if page > 0 else 0
                bookmarks.append((title, level, page_idx))
            return bookmarks
        except Exception as e:
            st.error(f"Error extracting bookmarks from {self.filename}: {e}")
            return []

    def identify_chapters_regex(self, text: str) -> List[Tuple[str, int, int]]:
        """Identifies chapters using improved regular expressions and infers level."""
        chapter_starts = []
        patterns = [
            (1, r"(?:^|\n)\s*Chapter\s+(\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # Chapter 1: Title or Chapter 1 Title
            (1, r"(?:^|\n)\s*Part\s+(\d+|[IVX]+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # Part I: Title or Part I Title
            (1, r"(?:^|\n)\s*Section\s+(\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # Section 1: Title
            (1, r"(?:^|\n)\s*\d+\.\s+([A-Z][^\n]+)"),  # 1. TITLE FORMAT
            (2, r"(?:^|\n)\s*(\d+\.\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # 1.1: Title or 1.1 Title
            (3, r"(?:^|\n)\s*(\d+\.\d+\.\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),  # 1.1.1: Title or 1.1.1 Title
            (2, r"(?:^|\n)\s*([A-Z]\.)\s+([^\n]+)"),  # A. Subtitle format
            (3, r"(?:^|\n)\s*([a-z]\.)\s+([^\n]+)")   # a. Sub-subtitle format
        ]
        
        for level, pattern in patterns:
            for match in re.finditer(pattern, text):
                title = match.group(0).strip()
                chapter_starts.append((title, level, match.start()))
        
        # Sort by position in text
        chapter_starts.sort(key=lambda x: x[2])
        return chapter_starts

    def find_page_number(self, char_index: int) -> int:
        """Finds the page number corresponding to a character index based on page character counts."""
        current_index = 0
        for page_num, char_count in enumerate(self.page_char_counts):
            if current_index <= char_index < current_index + char_count:
                return page_num
            current_index += char_count
        return self.document.page_count - 1  # Return last page if not found

    def process_bookmark_chunks(self, bookmarks: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
        """Process PDF using bookmarks to create content chunks."""
        chunks = []
        
        # Sort bookmarks by page number
        sorted_bookmarks = sorted(bookmarks, key=lambda x: x[2])
        
        for i, (title, level, start_page) in enumerate(sorted_bookmarks):
            # Determine end page - either before next bookmark at same/lower level or document end
            end_page = self.document.page_count - 1
            for j in range(i+1, len(sorted_bookmarks)):
                next_title, next_level, next_page = sorted_bookmarks[j]
                if next_level <= level:  # Same level or higher in hierarchy
                    end_page = next_page - 1
                    break
            
            # Extract content from the pages
            content = ""
            for pg in range(start_page, min(end_page + 1, self.document.page_count)):
                content += self.document.load_page(pg).get_text() + "\n"
            
            chunks.append({
                "title": title,
                "level": level,
                "start_page": start_page,
                "end_page": end_page,
                "content": content.strip()
            })
        
        return chunks

    def process_regex_chunks(self, chapter_starts: List[Tuple[str, int, int]], full_text: str) -> List[Dict[str, Any]]:
        """Process PDF using regex-identified chapters to create content chunks."""
        chunks = []
        
        for i, (title, level, start_pos) in enumerate(chapter_starts):
            # Find start page for this chunk
            start_page = self.find_page_number(start_pos)
            
            # Find end position and page
            if i + 1 < len(chapter_starts):
                end_pos = chapter_starts[i + 1][2]
                end_page = self.find_page_number(end_pos - 1)
            else:
                end_pos = len(full_text)
                end_page = self.document.page_count - 1
            
            content = full_text[start_pos:end_pos].strip()
            chunks.append({
                "title": title,
                "level": level,
                "start_page": start_page,
                "end_page": end_page,
                "content": content
            })
        
        return chunks

    def build_hierarchical_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize chunks into a hierarchical structure with parent-child relationships."""
        # Sort by level and then by start_page to ensure proper ordering
        chunks.sort(key=lambda x: (x["start_page"], x["level"]))
        
        # Add parent_id and children fields
        hierarchical_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["id"] = i
            chunk_copy["children"] = []
            hierarchical_chunks.append(chunk_copy)
        
        # Build parent-child relationships
        for i, chunk in enumerate(hierarchical_chunks):
            parent_id = None
            # Find the nearest previous chunk with a lower level
            for j in range(i-1, -1, -1):
                if hierarchical_chunks[j]["level"] < chunk["level"]:
                    parent_id = hierarchical_chunks[j]["id"]
                    hierarchical_chunks[j]["children"].append(i)
                    break
            chunk["parent_id"] = parent_id
        
        return hierarchical_chunks

    def process_pdf(self) -> Optional[Dict[str, Any]]:
        """Main function to process the PDF."""
        if not self.open_document():
            return None

        try:
            with st.spinner("Extracting text..."):
                full_text = self.extract_text_fitz()
                if not full_text:
                    st.error(f"Could not extract text from {self.filename}.")
                    return None

            with st.spinner("Processing document structure..."):
                # Extract Bookmarks (Table of Contents)
                bookmarks = self.extract_bookmarks_fitz()
                
                # Process based on bookmarks or regex
                if bookmarks:
                    st.info(f"Bookmarks found in {self.filename}. Using bookmarks for chapter identification.")
                    chunks = self.process_bookmark_chunks(bookmarks)
                else:
                    st.info(f"No bookmarks found in {self.filename}. Using pattern detection for chapter identification.")
                    chapter_starts = self.identify_chapters_regex(full_text)
                    chunks = self.process_regex_chunks(chapter_starts, full_text)
                
                # Build hierarchical structure
                hierarchical_chunks = self.build_hierarchical_chunks(chunks)
                
            return {
                "filename": self.filename,
                "chunks": chunks,
                "hierarchical_chunks": hierarchical_chunks,
                "bookmarks": bookmarks,
                "page_count": self.document.page_count
            }
        finally:
            self.close_document()

def display_hierarchical_toc(hierarchical_chunks):
    """Display a hierarchical table of contents without nested expanders."""
    # Extract all chunks and sort by page number and level
    sorted_chunks = sorted(hierarchical_chunks, key=lambda x: (x["start_page"], x["level"]))
    
    # Display TOC as a hierarchical list
    st.write("### Document Structure")
    
    # Create a dictionary for level-based indentation display
    for chunk in sorted_chunks:
        # Calculate the actual display level by counting parents
        display_level = 0
        parent_id = chunk["parent_id"]
        while parent_id is not None:
            display_level += 1
            parent = next((c for c in hierarchical_chunks if c["id"] == parent_id), None)
            if parent:
                parent_id = parent["parent_id"]
            else:
                parent_id = None
        
        # Create indentation with HTML spaces
        indent = "&nbsp;" * (display_level * 4)
        
        # Display the chunk with proper indentation
        st.markdown(f"{indent}📄 **{chunk['title']}** (Pages {chunk['start_page']+1}-{chunk['end_page']+1})", unsafe_allow_html=True)
    
    # Create a section for viewing chunk content
    st.write("---")
    st.subheader("View Chapter Content")
    
    # Create options for the dropdown with formatted titles
    chunk_options = [f"{c['title']} (Pages {c['start_page']+1}-{c['end_page']+1})" for c in sorted_chunks]
    selected_chunk_title = st.selectbox("Select a chapter to view:", chunk_options)
    
    # Find the selected chunk
    selected_index = chunk_options.index(selected_chunk_title)
    selected_chunk = sorted_chunks[selected_index]
    
    # Display the content of the selected chunk
    with st.expander(f"Content of {selected_chunk['title']}", expanded=True):
        st.text_area("Chapter text", selected_chunk['content'], height=300)

def export_chunks_to_csv(chunks, filename):
    """Export chunks to CSV file."""
    df = pd.DataFrame(chunks)
    # Select only the columns we want to export and create a copy to avoid warnings
    export_df = df[['title', 'level', 'start_page', 'end_page']].copy()
    # Add 1 to page numbers for display (convert from 0-indexed to 1-indexed)
    export_df.loc[:, 'start_page'] += 1
    export_df.loc[:, 'end_page'] += 1
    # Create a downloadable CSV
    return export_df.to_csv(index=False).encode('utf-8')

def export_chunks_to_markdown(chunks, filename):
    """Export chunks to Markdown file."""
    base_name = os.path.splitext(filename)[0]
    markdown_content = f"# {base_name} - Document Structure\n\n"
    
    for chunk in chunks:
        level_prefix = "#" * (chunk['level'] + 1)
        markdown_content += f"{level_prefix} {chunk['title']}\n\n"
        markdown_content += f"**Pages:** {chunk['start_page'] + 1}-{chunk['end_page'] + 1}\n\n"
        
        # Add a sample of content (first 500 characters)
        if chunk['content']:
            content_preview = chunk['content'][:500]
            if len(chunk['content']) > 500:
                content_preview += "..."
            markdown_content += f"{content_preview}\n\n"
        
        markdown_content += "---\n\n"
    
    return markdown_content.encode('utf-8')

def export_chunks_to_pdf_text(chunks, filename):
    """Export chunks structure to a text file (since actual PDF generation requires additional libraries)."""
    base_name = os.path.splitext(filename)[0]
    pdf_text_content = f"{base_name} - Document Structure\n"
    pdf_text_content += "=" * 50 + "\n\n"
    
    for chunk in chunks:
        level_indent = "  " * (chunk['level'] - 1)
        pdf_text_content += f"{level_indent}{chunk['title']}\n"
        pdf_text_content += f"{level_indent}Pages: {chunk['start_page'] + 1}-{chunk['end_page'] + 1}\n"
        pdf_text_content += f"{level_indent}{'-' * 30}\n"
        
        # Add a sample of content
        if chunk['content']:
            content_preview = chunk['content'][:300]
            if len(chunk['content']) > 300:
                content_preview += "..."
            pdf_text_content += f"{level_indent}{content_preview}\n"
        
        pdf_text_content += "\n"
    
    return pdf_text_content.encode('utf-8')

def create_teaching_interface(result, api_key):
    """Create interactive teaching interface with LangGraph"""
    st.subheader("🤖 AI Teacher - Ask Questions About Your Book")
    
    if not RAG_AVAILABLE:
        st.error("RAG features are not available. Please install required packages.")
        st.code("pip install langchain-groq langgraph sentence-transformers chromadb")
        return
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("Setting up AI teacher..."):
            try:
                st.session_state.rag_system = BookTeachingRAG()
                st.session_state.rag_system.index_book_content(result['chunks'])
                st.success("AI Teacher initialized! 🎓")
            except Exception as e:
                st.error(f"Failed to initialize AI teacher: {str(e)}")
                return
    
    # Model configuration
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox(
            "Choose AI Model:", 
            SUPPORTED_MODELS,
            index=0
        )
    
    with col2:
        chapter_titles = [chunk['title'] for chunk in result['chunks']]
        selected_chapter = st.selectbox(
            "Focus on chapter (optional):",
            ["All Chapters"] + chapter_titles
        )
    
    # Setup Groq model if not already done
    if not st.session_state.rag_system.groq_model:
        if api_key:
            try:
                st.session_state.rag_system.setup_groq_model(api_key, selected_model)
                st.success("AI Teacher ready! 🎓")
            except Exception as e:
                st.error(f"Failed to setup AI model: {str(e)}")
                st.info("Please check your Groq API key and try again.")
                return
        else:
            st.warning("Please enter your Groq API key in the sidebar to start teaching!")
            st.info("Get your free API key from: https://console.groq.com/")
            return
    
    # Initialize chat state
    if 'teaching_messages' not in st.session_state:
        st.session_state.teaching_messages = []
    
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = "book_teaching_session"
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.teaching_messages = []
        st.rerun()
    
    # Display chat history
    for message in st.session_state.teaching_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"• {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about this book..."):
        # Add user message
        st.session_state.teaching_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Teaching..."):
                chapter_filter = selected_chapter if selected_chapter != "All Chapters" else None
                
                # Convert messages to LangChain format
                lc_messages = []
                for msg in st.session_state.teaching_messages[:-1]:  # Exclude the current user message
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        lc_messages.append(AIMessage(content=msg["content"]))
                
                try:
                    response = st.session_state.rag_system.teach_topic(
                        prompt, 
                        lc_messages, 
                        chapter_filter,
                        st.session_state.thread_id
                    )
                    
                    ai_content = response["response"].content
                    st.markdown(ai_content)
                    
                    # Show sources
                    if response["sources"]:
                        with st.expander("📚 Sources Used"):
                            for source in response["sources"]:
                                st.write(f"• {source}")
                    
                    # Add learning suggestions
                    with st.expander("🎯 Learning Suggestions", expanded=False):
                        st.markdown("""
                        **To enhance your learning:**
                        - Try asking follow-up questions about specific concepts
                        - Request examples or real-world applications
                        - Ask for connections to other chapters
                        - Request summaries of complex topics
                        - Ask "How does this relate to...?" for deeper understanding
                        """)
                    
                    # Add to history
                    st.session_state.teaching_messages.append({
                        "role": "assistant", 
                        "content": ai_content,
                        "sources": response["sources"]
                    })
                    
                except Exception as e:
                    st.error(f"Teaching error: {str(e)}")
                    st.info("Please try again or check your API key.")

def main():
    st.title("📚 Book AI Processor with Teaching Assistant")
    st.write("Upload a PDF and get an AI teacher powered by Groq!")
    
    # Note about warnings (optional to display)
    # st.info("ℹ️ Note: PyTorch warnings about 'torch.classes' are harmless and can be ignored.")
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input(
            "Groq API Key:", 
            type="password", 
            placeholder="Enter your Groq API key",
            help="Get your free API key from https://console.groq.com/"
        )
        
        if api_key:
            st.success("API Key configured! 🎉")
        
        st.markdown("---")
        st.markdown("### 🚀 Features")
        st.markdown("""
        - **Document Structure**: Extract chapters and sections
        - **Smart Search**: Find content across the book
        - **AI Teacher**: Interactive learning with RAG
        - **Export Options**: Download structured data
        """)
    
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📖 Structure", "🔍 Search", "🎓 AI Teacher", "📊 Export"])
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name
        
        try:
            with st.spinner("Processing PDF..."):
                processor = PDFProcessor(temp_file_path)
                result = processor.process_pdf()
            
            if result:
                with tab1:
                    st.success(f"✅ Processing complete: {result['filename']} ({result['page_count']} pages)")
                    
                    # Display hierarchical table of contents
                    st.subheader("Document Structure")
                    display_hierarchical_toc(result["hierarchical_chunks"])
                
                with tab2:
                    st.subheader("Search Document")
                    search_query = st.text_input("Enter search term:")
                    if search_query:
                        st.write(f"Searching for: '{search_query}'")
                        results = []
                        for chunk in result["chunks"]:
                            if search_query.lower() in chunk["content"].lower():
                                results.append(chunk)
                        
                        if results:
                            st.write(f"Found {len(results)} matches:")
                            for r in results:
                                with st.expander(f"{r['title']} (Pages {r['start_page']+1}-{r['end_page']+1})"):
                                    # Highlight matches in content preview
                                    content_preview = r["content"][:500]
                                    st.markdown(content_preview.replace(search_query, f"**{search_query}**"), unsafe_allow_html=True)
                        else:
                            st.info("No matches found.")
                
                with tab3:
                    create_teaching_interface(result, api_key)
                  with tab4:
                    # Show supported export formats in subheader
                    st.subheader("Export Options: CSV · PDF · MD")
                    
                    st.write("Choose your preferred export format:")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv_data = export_chunks_to_csv(result["chunks"], result["filename"])
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv_data,
                            file_name=f"{os.path.splitext(result['filename'])[0]}_structure.csv",
                            mime="text/csv",
                            help="Download structured data as CSV for spreadsheet analysis"
                        )
                    
                    with col2:
                        pdf_data = export_chunks_to_pdf_text(result["chunks"], result["filename"])
                        st.download_button(
                            label="📄 Download PDF Text",
                            data=pdf_data,
                            file_name=f"{os.path.splitext(result['filename'])[0]}_structure.txt",
                            mime="text/plain",
                            help="Download document structure as formatted text"
                        )
                    
                    with col3:
                        md_data = export_chunks_to_markdown(result["chunks"], result["filename"])
                        st.download_button(
                            label="📝 Download Markdown",
                            data=md_data,
                            file_name=f"{os.path.splitext(result['filename'])[0]}_structure.md",
                            mime="text/markdown",
                            help="Download as Markdown file for documentation"
                        )
            else:
                st.error("Processing failed.")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {e}")

if __name__ == "__main__":
    main()