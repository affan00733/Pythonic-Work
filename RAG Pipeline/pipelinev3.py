import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Step 1: Load PDF Documents
def load_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = UnstructuredPDFLoader(os.path.join(pdf_folder, filename))
            documents.extend(loader.load())
    return documents

# Step 2: Split Documents into Chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# Step 3: Generate Embeddings
def generate_embeddings(split_docs):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model, embedding_model.embed_documents([doc.page_content for doc in split_docs])

# Step 4: Store Embeddings in a Vector Store
def create_vector_store(split_docs, embedding_model):
    return Chroma.from_documents(split_docs, embedding_model)

# Step 5: Initialize RAG Pipeline with Local Model
def initialize_rag_pipeline(vector_store, model_name="tiiuae/falcon-7b-instruct"):
    retriever = vector_store.as_retriever()

    # Load Local Model for Text Generation
    print("Loading local language model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    # Define Local Completion Function
    def local_completion(prompt):
        response = generator(prompt, max_length=200, num_return_sequences=1, do_sample=True)
        return response[0]["generated_text"]

    # Define the RAG pipeline function
    def rag_pipeline(query):
        retrieved_docs = retriever.get_relevant_documents(query)
        combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Query: {query}\nContext: {combined_context}\nAnswer:"
        return local_completion(prompt)

    return rag_pipeline

# Main Function
if __name__ == "__main__":
    # Directory containing PDF files
    pdf_folder = "sample_pdfs"

    print("Loading PDF documents...")
    documents = load_pdfs(pdf_folder)

    print("Splitting documents into chunks...")
    split_docs = split_documents(documents)

    print("Generating embeddings...")
    embedding_model, _ = generate_embeddings(split_docs)

    print("Creating vector store...")
    vector_store = create_vector_store(split_docs, embedding_model)

    print("Initializing RAG pipeline...")
    rag_pipeline = initialize_rag_pipeline(vector_store, model_name="tiiuae/falcon-7b-instruct")

    # Example Query
    user_query = "What are the main topics discussed in the documents?"
    print("Querying the RAG pipeline...")
    response = rag_pipeline(user_query)

    print("Generated Response:")
    print(response)
