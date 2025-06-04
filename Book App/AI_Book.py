
import os
import re
import tempfile
import argparse
import sys
import pandas as pd
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional, Any, Sequence

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
        self.groq_model = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.7
        )
        self._initialize_workflow()

    def _initialize_workflow(self):
        teaching_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI teacher specializing in adaptive learning from books. Your teaching follows a structured pedagogical approach.\n\n## Teaching Philosophy & Pattern:\n\n### 🎯 **Learning Objective Identification**\n- First understand what the student wants to learn\n- Identify their current knowledge level\n- Set clear, achievable learning goals\n\n### 📚 **Structured Teaching Pattern**\nWhen teaching any concept, follow this pattern:\n\n1. **FOUNDATION** 🏗️\n   - Start with core definitions and basic concepts\n   - Provide clear, simple explanations\n   - Use analogies from everyday life\n\n2. **CONTEXT** 🌍\n   - Explain where this fits in the bigger picture\n   - Connect to previously learned material\n   - Show relevance and importance\n\n3. **EXAMPLES** 💡\n   - Provide concrete, relatable examples\n   - Use case studies from the book content\n   - Show practical applications\n\n4. **ANALYSIS** 🔍\n   - Break down complex ideas into components\n   - Explain cause-and-effect relationships\n   - Highlight patterns and principles\n\n5. **APPLICATION** 🚀\n   - Suggest how to apply this knowledge\n   - Provide practice scenarios\n   - Connect to real-world situations\n\n6. **REINFORCEMENT** 🎯\n   - Summarize key takeaways\n   - Suggest follow-up questions for deeper learning\n   - Recommend related topics to explore\n\n### 💬 **Communication Style**\n- **Clarity**: Use simple, clear language\n- **Engagement**: Ask thought-provoking questions\n- **Encouragement**: Maintain positive, supportive tone\n- **Adaptation**: Adjust complexity based on student responses\n- **Source-Grounded**: Always reference the book content\n\n### 🔄 **Interactive Learning**\n- Ask \"Do you understand?\" or \"What would you like to explore further?\"\n- Encourage questions and clarification requests\n- Provide multiple perspectives on complex topics\n- Use Socratic method when appropriate\n\n### 📖 **Content Integration**\n- Always ground explanations in the provided book context\n- Reference specific chapters and page numbers\n- Quote relevant passages when helpful\n- Maintain fidelity to the author's intent\n\n### 🎓 **Assessment & Progress**\n- Check understanding with gentle questioning\n- Provide positive reinforcement for engagement\n- Suggest next steps in the learning journey\n- Identify knowledge gaps and address them\n\nRemember: You are not just answering questions - you are facilitating deep, meaningful learning experiences based on the book's content."""),
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
        context_data = self.retrieve_context(user_question, selected_chapter)
        context_text = ""
        source_info = []
        for i, doc in enumerate(context_data['documents'][0]):
            if i < len(context_data['metadatas'][0]):
                metadata = context_data['metadatas'][0][i]
                context_text += f"\n--- Context {i+1} ---\n{doc}\n"
                source_info.append(f"Chapter: {metadata['chapter']}, Pages: {metadata['start_page']}-{metadata['end_page']}")
        state = {
            "messages": messages_history + [HumanMessage(content=user_question)],
            "context": context_text,
            "sources": source_info
        }
        config = {"configurable": {"thread_id": thread_id}}
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
        self.page_char_counts = []

    def open_document(self) -> bool:
        try:
            if self.file_extension == ".pdf":
                self.document = fitz.open(self.file_path)
                return True
            else:
                raise ValueError(f"Unsupported file type: {self.file_extension}")
        except Exception as e:
            print(f"Error opening {self.filename}: {e}")
            return False

    def close_document(self) -> None:
        if self.document:
            self.document.close()
            self.document = None

    def extract_text_fitz(self) -> str:
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
            print(f"Error extracting text from {self.filename}: {e}")
            return ""

    def extract_bookmarks_fitz(self) -> List[Tuple[str, int, int]]:
        try:
            if not self.document:
                raise ValueError("Document not open.")
            toc = self.document.get_toc()
            if not toc:
                return []
            bookmarks = []
            for level, title, page in toc:
                page_idx = page - 1 if page > 0 else 0
                bookmarks.append((title, level, page_idx))
            return bookmarks
        except Exception as e:
            print(f"Error extracting bookmarks from {self.filename}: {e}")
            return []

    def identify_chapters_regex(self, text: str) -> List[Tuple[str, int, int]]:
        chapter_starts = []
        patterns = [
            (1, r"(?:^|\n)\s*Chapter\s+(\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (1, r"(?:^|\n)\s*Part\s+(\d+|[IVX]+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (1, r"(?:^|\n)\s*Section\s+(\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (1, r"(?:^|\n)\s*\d+\.\s+([A-Z][^\n]+)"),
            (2, r"(?:^|\n)\s*(\d+\.\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (3, r"(?:^|\n)\s*(\d+\.\d+\.\d+)(?:\s*[:.-]\s*|\s+)([^\n]+)?"),
            (2, r"(?:^|\n)\s*([A-Z]\.)\s+([^\n]+)"),
            (3, r"(?:^|\n)\s*([a-z]\.)\s+([^\n]+)")
        ]
        for level, pattern in patterns:
            for match in re.finditer(pattern, text):
                title = match.group(0).strip()
                chapter_starts.append((title, level, match.start()))
        chapter_starts.sort(key=lambda x: x[2])
        return chapter_starts

    def find_page_number(self, char_index: int) -> int:
        current_index = 0
        for page_num, char_count in enumerate(self.page_char_counts):
            if current_index <= char_index < current_index + char_count:
                return page_num
            current_index += char_count
        return self.document.page_count - 1

    def process_bookmark_chunks(self, bookmarks: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
        chunks = []
        sorted_bookmarks = sorted(bookmarks, key=lambda x: x[2])
        for i, (title, level, start_page) in enumerate(sorted_bookmarks):
            end_page = self.document.page_count - 1
            for j in range(i+1, len(sorted_bookmarks)):
                next_title, next_level, next_page = sorted_bookmarks[j]
                if next_level <= level:
                    end_page = next_page - 1
                    break
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
        chunks = []
        for i, (title, level, start_pos) in enumerate(chapter_starts):
            start_page = self.find_page_number(start_pos)
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
        chunks.sort(key=lambda x: (x["start_page"], x["level"]))
        hierarchical_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["id"] = i
            chunk_copy["children"] = []
            hierarchical_chunks.append(chunk_copy)
        for i, chunk in enumerate(hierarchical_chunks):
            parent_id = None
            for j in range(i-1, -1, -1):
                if hierarchical_chunks[j]["level"] < chunk["level"]:
                    parent_id = hierarchical_chunks[j]["id"]
                    hierarchical_chunks[j]["children"].append(i)
                    break
            chunk["parent_id"] = parent_id
        return hierarchical_chunks

    def process_pdf(self) -> Optional[Dict[str, Any]]:
        if not self.open_document():
            return None
        try:
            full_text = self.extract_text_fitz()
            if not full_text:
                print(f"Could not extract text from {self.filename}.")
                return None
            bookmarks = self.extract_bookmarks_fitz()
            if bookmarks:
                print(f"Bookmarks found in {self.filename}. Using bookmarks for chapter identification.")
                chunks = self.process_bookmark_chunks(bookmarks)
            else:
                print(f"No bookmarks found in {self.filename}. Using pattern detection for chapter identification.")
                chapter_starts = self.identify_chapters_regex(full_text)
                chunks = self.process_regex_chunks(chapter_starts, full_text)
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

def export_chunks_to_csv(chunks, filename):
    df = pd.DataFrame(chunks)
    export_df = df[['title', 'level', 'start_page', 'end_page']]
    export_df['start_page'] += 1
    export_df['end_page'] += 1
    export_df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Exported structure to {filename}")

def print_hierarchical_toc(hierarchical_chunks):
    sorted_chunks = sorted(hierarchical_chunks, key=lambda x: (x["start_page"], x["level"]))
    for chunk in sorted_chunks:
        display_level = 0
        parent_id = chunk["parent_id"]
        while parent_id is not None:
            display_level += 1
            parent = next((c for c in hierarchical_chunks if c["id"] == parent_id), None)
            if parent:
                parent_id = parent["parent_id"]
            else:
                parent_id = None
        indent = "    " * display_level
        print(f"{indent}- {chunk['title']} (Pages {chunk['start_page']+1}-{chunk['end_page']+1})")

def main():
    parser = argparse.ArgumentParser(description="Book AI Processor (CLI)")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--api_key", help="Groq API Key", required=False)
    parser.add_argument("--model", help="Groq Model", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS[0])
    parser.add_argument("--export_csv", help="Export structure to CSV file", default=None)
    parser.add_argument("--teach", help="Ask a question to the AI teacher", default=None)
    parser.add_argument("--chapter", help="Focus on chapter title (optional)", default=None)
    args = parser.parse_args()

    processor = PDFProcessor(args.pdf)
    result = processor.process_pdf()
    if not result:
        print("Processing failed.")
        sys.exit(1)

    print(f"\nProcessed: {result['filename']} ({result['page_count']} pages)")
    print("\nTable of Contents:")
    print_hierarchical_toc(result["hierarchical_chunks"])

    if args.export_csv:
        export_chunks_to_csv(result["chunks"], args.export_csv)

    if args.teach:
        if not RAG_AVAILABLE:
            print("RAG features are not available. Please install required packages:")
            print("pip install langchain-groq langgraph sentence-transformers chromadb")
            sys.exit(1)
        if not args.api_key:
            print("Please provide your Groq API key with --api_key.")
            sys.exit(1)
        rag_system = BookTeachingRAG()
        rag_system.index_book_content(result['chunks'])
        rag_system.setup_groq_model(args.api_key, args.model)
        lc_messages = []
        chapter_filter = args.chapter if args.chapter else None
        response = rag_system.teach_topic(
            args.teach,
            lc_messages,
            chapter_filter,
            thread_id="cli_session"
        )
        print("\nAI Teacher Response:\n")
        print(response["response"].content)
        if response["sources"]:
            print("\nSources:")
            for source in response["sources"]:
                print(f"- {source}")

if __name__ == "__main__":
    main()
