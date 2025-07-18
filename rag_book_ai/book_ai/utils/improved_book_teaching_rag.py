from typing import List, Dict, Tuple, Optional, Any, Sequence
import chromadb
from typing_extensions import Annotated, TypedDict
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from sentence_transformers import SentenceTransformer

class BookTeachingRAG:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = None
        self.groq_model = None
        self.app = None
        
    def _split_into_paragraphs(self, text):
        """Split text into paragraphs for better semantic chunking"""
        # First split by double newlines (typical paragraph breaks)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # For paragraphs that are still too long, try to split at single newlines
        result = []
        for para in paragraphs:
            if len(para.split()) > 200:  # If paragraph is very long
                subparas = [sp.strip() for sp in para.split('\n') if sp.strip()]
                result.extend(subparas)
            else:
                result.append(para)
                
        return result
    
    def _extract_section_header(self, text):
        """Extract a possible section header from text to improve metadata"""
        # Look for common header patterns
        header_patterns = [
            r"^#+\s+(.+)$",                     # Markdown headers
            r"^(\d+\.[\d\.]*\s+[A-Z][^\.]+)",   # Numbered sections like "1.2 Title"
            r"^([A-Z][^\.]{3,50})\s*$",         # UPPERCASE or Title Case headers
            r"^(Chapter \d+[:\s]+[^\n]+)",      # Chapter headers
            r"^(Section \d+[:\s]+[^\n]+)"       # Section headers
        ]
        
        lines = text.split('\n')
        for line in lines[:5]:  # Check only first 5 lines for headers
            line = line.strip()
            for pattern in header_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    return match.group(1)
        
        return None
    
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
            ("system", """I want you to act as an expert tutor and write a "chapter" on the topic I specify. Use very clear, simple language so a beginner can follow. Structure your response as follows:

1. Title and Introduction
   - Give a short, friendly chapter title.
   - Explain in a sentence why this topic matters or how it can help the learner.

2. Learning Objectives
   - List 3-5 things the learner will understand or be able to do after reading.

3. Background & Context
   - Briefly describe where this topic fits in the bigger picture.
   - Define any basic terms or ideas the learner needs to know first.

4. Key Concepts (Broken into Sections)
   - Divide the topic into logical sections or steps.
   - For each section:
     • Give a clear heading.
     • Explain the core idea in simple words.
     • Show a concrete example or analogy.
     • If useful, suggest a simple "visual" (e.g., "imagine …") or a mental picture.

5. Step-by-Step Explanations or Process
   - If the topic involves procedures or stages, list them one by one.
   - Explain each stage simply, why it matters, and what to watch out for.

6. Real-World Applications or Use Cases
   - Describe 1-2 simple scenarios where this knowledge applies.
   - Show how it could be used in everyday life or a project.

7. Common Mistakes or FAQs
   - Point out pitfalls or misunderstandings beginners often have.
   - Provide short Q&A: e.g., "Q: Is X always true? A: Not always, because…"

8. Summary
   - Restate the main points in a few bullet lines.
   - Remind the learner what they should now understand or do.

9. Practice or Reflection
   - Give 1-3 simple exercises, thought questions, or small tasks to try.
   - Encourage the learner to reflect: "How would you apply this? What challenges might arise?"

10. Further Resources (Optional)
   - Suggest 1-3 next steps: keywords to search, book chapters, tutorials, or simple tools to explore.

Remember to always base your teaching on the book content. Use specific examples, ideas, and explanations from the book when creating your chapters."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        class TeachingState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            context: str
            sources: list
        
        workflow = StateGraph(state_schema=TeachingState)
        
        def call_teaching_model(state: TeachingState):
            context_message = f"\n--- BOOK CONTEXT ---\n{state.get('context', '')}\n--- END CONTEXT ---\n"
            
            messages_with_context = list(state["messages"])
            if messages_with_context and isinstance(messages_with_context[-1], HumanMessage):
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
        """Index book content by chapter for retrieval using improved chunking"""
        # Create collection for this book if it doesn't exist
        collection_name = f"book_collection_{hash(str(book_chunks))}"
        
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
            
            # Create improved chunks using semantic boundaries
            all_chunks = self.create_rag_chunks(book_chunks)
            
            ids = []
            texts = []
            metadatas = []
            
            for i, chunk in enumerate(all_chunks):
                ids.append(f"chunk_{i}")
                texts.append(chunk['text'])
                metadatas.append({
                    "title": chunk['title'],
                    "start_page": chunk['start_page'],
                    "end_page": chunk['end_page'],
                    "level": chunk['level'],
                    "section_header": chunk.get('section_header', ''),
                })
            
            # Create embeddings and add to collection
            embeddings = self.embedding_model.encode(texts).tolist()
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        
    def create_rag_chunks(self, book_chunks):
        """Split large chapters into smaller, contextual chunks using semantic boundaries and overlapping windows"""
        rag_chunks = []

        # Adjusted parameters for better semantic chunking
        max_window_size = 500  # max words per chunk (increased for more context)
        min_window_size = 300  # minimum words to maintain context
        overlap_size = 100     # increased overlap for better continuity
        
        for chapter in book_chunks:
            # Try to identify paragraphs for more natural chunking
            paragraphs = self._split_into_paragraphs(chapter['content'])
            
            # Track accumulated content to ensure chunks aren't too small
            current_chunk = []
            current_word_count = 0
            
            for paragraph in paragraphs:
                paragraph_words = paragraph.split()
                paragraph_word_count = len(paragraph_words)
                
                # If adding this paragraph exceeds our max window size and we already have content
                if current_word_count + paragraph_word_count > max_window_size and current_word_count >= min_window_size:
                    # Create a chunk from accumulated paragraphs
                    chunk_text = ' '.join(current_chunk)
                    
                    # Add section identifier if possible by looking for headers
                    section_header = self._extract_section_header(chunk_text)
                    
                    rag_chunks.append({
                        'text': chunk_text,
                        'title': chapter['title'],
                        'start_page': chapter['start_page'],
                        'end_page': chapter['end_page'],
                        'level': chapter['level'],
                        'section_header': section_header or "Section in " + chapter['title'],
                    })
                    
                    # Reset with overlap - keep some paragraphs for context
                    overlap_words = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else ''
                    current_chunk = [overlap_words] if overlap_words else []
                    current_word_count = len(overlap_words.split()) if overlap_words else 0
                
                # Add current paragraph to the chunk being built
                current_chunk.append(paragraph)
                current_word_count += paragraph_word_count
            
            # Don't forget to add the last chunk if there's content left
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                section_header = self._extract_section_header(chunk_text)
                
                rag_chunks.append({
                    'text': chunk_text,
                    'title': chapter['title'],
                    'start_page': chapter['start_page'],
                    'end_page': chapter['end_page'],
                    'level': chapter['level'],
                    'section_header': section_header or "Section in " + chapter['title'],
                })

        return rag_chunks
    
    def retrieve_context(self, query, chapter_filter=None):
        """Retrieve most relevant chunks for the query"""
        if not self.collection:
            return {"documents": [[]], "metadatas": [[]]}
        
        query_embedding = self.embedding_model.encode(query)
        
        where_clause = None
        if chapter_filter:
            where_clause = {"title": {"$eq": chapter_filter}}
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,  # Increased for better context
            where=where_clause
        )
        
        return results
    
    def teach_topic(self, user_question, messages_history, selected_chapter=None, thread_id="default"):
        """Main teaching function using RAG and LangGraph"""
        context_data = self.retrieve_context(user_question, selected_chapter)
        
        context_text = ""
        source_info = []
        
        for i, doc in enumerate(context_data['documents'][0]):
            if i < len(context_data['metadatas'][0]):
                metadata = context_data['metadatas'][0][i]
                context_text += f"\n--- Context {i+1} ---\n{doc}\n"
                chapter_title = metadata.get('title', 'Unknown Chapter')
                section_header = metadata.get('section_header', '')
                pages = f"Pages: {metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')}"
                
                source_desc = f"Chapter: {chapter_title}"
                if section_header and section_header != f"Section in {chapter_title}":
                    source_desc += f" - {section_header}"
                source_desc += f", {pages}"
                
                source_info.append(source_desc)
        
        state = {
            "messages": messages_history + [HumanMessage(content=user_question)],
            "context": context_text,
            "sources": source_info,
            "chapter": selected_chapter
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            output = self.app.invoke(state, config)
            ai_response = output["messages"][-1]
            
            return {
                "response": ai_response,
                "sources": source_info,
                "context_used": len(context_data['documents'][0])
            }
        except Exception as e:
            if "401" in str(e) and "invalid_api_key" in str(e):
                raise ValueError("Your Groq API key appears to be invalid or expired. Please check your API key.")
            raise e
