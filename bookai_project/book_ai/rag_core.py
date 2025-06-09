# book_ai/rag_core.py

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_book(file_path):
    """Loads the book text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_vector_store(text):
    """Creates a FAISS vector store from the text."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Use a free sentence transformer model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def create_qa_chain(vector_store):
    """Creates a question-answering chain."""
    # For a completely free experience, we can use a smaller, local model.
    # Note: Performance will be less than larger models.
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.7,
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain

# Load the book and create the vector store and QA chain once when the server starts
book_text = load_book('book.txt')
vector_store = create_vector_store(book_text)
qa_chain = create_qa_chain(vector_store)

def get_answer(query):
    """Gets an answer from the QA chain."""
    return qa_chain.run(query)