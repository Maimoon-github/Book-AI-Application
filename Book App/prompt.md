You are tasked with building a Book-AI-Application that can import a PDF or book-related document, identify its topics or chapters, and break them into separate “chunks” with their page information. Follow these steps:

1. **Document Import**

   * Create a file picker so users can upload documents (PDFs, and optionally EPUB, MOBI, DOCX, or TXT).
   * Decide if you’ll store the uploads temporarily (just for processing) or permanently (for later access).

2. **Content Extraction**

   * For PDFs, use a library like PyPDF2, pdfminer.six, or PyMuPDF (fitz) to extract text and metadata.
   * If possible, extract bookmarks or a table of contents (ToC) from the PDF; these often list chapter titles and start pages.
   * Optionally, analyze fonts or layout (font size, bold/italic) to find headings if no ToC exists.

3. **Topic/Chapter Identification**

   * **If you have a ToC/bookmarks:** Each entry gives you a chapter title and the page where it starts. The chapter ends right before the next one starts (or at the end of the document).
   * **If no ToC/bookmarks:**

     * Scan each page’s text and look for patterns like “Chapter X”, “Part X”, or other consistent headings using regular expressions.
     * You can also inspect font sizes or positions (if your library provides that) to detect titles.
   * (Advanced) If there are no clear markers, you could use NLP methods (e.g., topic modeling) to find where the subject changes—though this is more complex.

4. **Content Chunking and Page Association**

   * Once you know each chapter’s start page (and where it ends), extract text for that range of pages.
   * If a chapter starts or ends mid-page, use layout info (coordinates) to get only the relevant portion.
   * For each chunk, save:

     * Chapter title
     * Chunk text
     * Starting page number
     * Ending page number

5. **Presenting the Chunks**

   * In your app’s UI, list all identified chapters.
   * When a user selects a chapter, show its text and clearly display the page range.
   * Allow easy navigation between chunks, and optionally add a search feature (within a chunk or across all chunks).

**Example Pseudocode (Python-like):**

```
# Assume 'uploaded_pdf_file' is the path to the user’s PDF

# Step 2: Extract basic structure
pdf = open_pdf(uploaded_pdf_file)  # e.g., using fitz or PyPDF2
chapters = extract_bookmarks(pdf)  # [(title, start_page), ...]

# Step 3: Identify chapters
if chapters:
    identified = chapters
else:
    all_text = {}
    for pg in range(pdf.page_count):
        text = extract_text_from_page(pdf, pg)
        all_text[pg] = text
    identified = find_chapter_starts(all_text)  # custom function

# Determine end pages
sections = []
for i, (title, start) in enumerate(identified):
    end = pdf.page_count - 1
    if i + 1 < len(identified):
        end = identified[i+1][1] - 1
    sections.append((title, start, end))

# Step 4: Chunk content
chunks = []
for title, start, end in sections:
    text = ""
    for pg in range(start, end + 1):
        text += extract_text_from_page(pdf, pg) + "\n"
    chunks.append({
        "title": title,
        "content": text.strip(),
        "start_page": start,
        "end_page": end
    })

close_pdf(pdf)
# Step 5 is handled in your UI code
```

**Key Points to Remember**

* Use Python (e.g., PyPDF2, pdfminer.six, or PyMuPDF) for PDF parsing.
* Scanned PDFs need OCR (e.g., Tesseract) before you can extract text.
* The success of chapter detection depends on how well-structured the PDF is.
* Design your UI so users can easily select chapters, read chunks, and see page info.
* Implement robust error handling for encrypted or corrupted files.

Good luck! 👍


````
## Next Step: Making Your Book AI RAG-Based for Teaching 🤖📚

Now that you have a solid foundation for extracting and chunking book content, let's transform it into an intelligent teaching system using RAG (Retrieval-Augmented Generation) with modern LangGraph and Groq integration.

### Step 6: RAG Architecture Setup 🏗️

#### 6.1: Dependencies Installation
```bash
pip install streamlit chromadb sentence-transformers langchain-groq langgraph langchain-core typing-extensions
````

#### 6.2: Enhanced RAG System with LangGraph

```python
import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

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
            ("system", """You are an expert AI teacher helping students learn from books. 
            
            Your teaching style:
            🎓 **Educational Excellence**: Provide clear, comprehensive explanations
            📚 **Source-Based**: Always reference the book content provided
            💡 **Engaging**: Use examples, analogies, and interactive elements
            🔍 **Deep Learning**: Connect concepts to broader themes
            ❓ **Curiosity Builder**: Suggest follow-up questions
            
            When responding:
            1. Give detailed explanations based on the book context
            2. Define key terms and concepts
            3. Provide examples when relevant
            4. Show how topics connect to bigger ideas
            5. Encourage deeper exploration with questions
            6. Reference specific chapters and pages when possible
            
            Keep responses educational, engaging, and grounded in the source material."""),
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
        """Split large chapters into smaller, contextual chunks"""
        rag_chunks = []
        
        for chapter in book_chunks:
            chapter_text = chapter['content']
            sentences = chapter_text.split('. ')
            
            current_chunk = ""
            word_count = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                if word_count + sentence_words > 400:
                    if current_chunk.strip():
                        rag_chunks.append({
                            'text': current_chunk.strip(),
                            'metadata': {
                                'chapter': chapter['title'],
                                'start_page': chapter['start_page'],
                                'end_page': chapter['end_page'],
                                'chunk_type': 'content'
                            }
                        })
                    current_chunk = sentence + '. '
                    word_count = sentence_words
                else:
                    current_chunk += sentence + '. '
                    word_count += sentence_words
            
            if current_chunk.strip():
                rag_chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'chapter': chapter['title'],
                        'start_page': chapter['start_page'],
                        'end_page': chapter['end_page'],
                        'chunk_type': 'content'
                    }
                })
        
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
```

### Step 7: Enhanced Streamlit Teaching Interface 💬

```python
def create_teaching_interface(result, api_key):
    """Create interactive teaching interface with LangGraph"""
    st.subheader("🤖 AI Teacher - Ask Questions About Your Book")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = BookTeachingRAG()
        with st.spinner("Setting up AI teacher..."):
            st.session_state.rag_system.index_book_content(result['chunks'])
    
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
            st.session_state.rag_system.setup_groq_model(api_key, selected_model)
        else:
            st.error("Please enter your Groq API key to start teaching!")
            return
    
    # Initialize chat state
    if 'teaching_messages' not in st.session_state:
        st.session_state.teaching_messages = []
    
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = "book_teaching_session"
    
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
                    
                    # Add to history
                    st.session_state.teaching_messages.append({
                        "role": "assistant", 
                        "content": ai_content,
                        "sources": response["sources"]
                    })
                    
                except Exception as e:
                    st.error(f"Teaching error: {str(e)}")

# Update your main function to include the teaching interface
def main():
    st.title("📚 Book AI Processor with Teaching Assistant")
    st.write("Upload a PDF and get an AI teacher powered by Groq!")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Groq API Key:", 
        type="password", 
        placeholder="Enter your Groq API key"
    )
    
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Structure", "Search", "🎓 AI Teacher", "Export"])
        
        # ... existing processing code ...
        
        if result:  # Assuming 'result' contains your processed book data
            with tab3:
                create_teaching_interface(result, api_key)
            
            # ... existing tabs code ...
```

### Step 8: Key Improvements & Benefits 🚀

#### Modern Architecture Features:

* **LangGraph Integration**: Advanced conversation flow management
* **Groq API**: Fast and efficient language model inference
* **Memory Management**: Persistent conversation context
* **Source Attribution**: Every response linked to book content
* **Flexible Model Selection**: Choose from multiple Groq models

#### Enhanced Teaching Capabilities:

* **Contextual Responses**: AI grounded in actual book content
* **Conversation Memory**: Remembers previous questions and context
* **Chapter-Specific Focus**: Can concentrate on specific sections
* **Educational Prompting**: Optimized for learning and teaching

#### User Experience Improvements:

* **Real-time Processing**: Fast responses with Groq models
* **Source Transparency**: Users can verify information
* **Interactive Learning**: Natural conversation flow
* **Error Handling**: Robust error management

Your Book AI application now leverages cutting-edge RAG architecture with LangGraph and Groq to provide an intelligent, context-aware teaching assistant that transforms any book into an interactive learning experience! 📖✨
