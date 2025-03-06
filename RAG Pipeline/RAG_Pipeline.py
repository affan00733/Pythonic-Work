import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fitz  # PyMuPDF for PDF text extraction

class RAGPipeline:
    def __init__(self, pdf_folder: str, model_name: str = "facebook/bart-large"):
        self.pdf_folder = pdf_folder
        self.documents = self._load_documents()
        self.vectorizer = TfidfVectorizer()
        self.doc_embeddings = self.vectorizer.fit_transform(self.documents)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text

    def _load_documents(self) -> list:
        documents = []
        for file_name in os.listdir(self.pdf_folder):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, file_name)
                text = self._extract_text_from_pdf(pdf_path)
                documents.append(text)
        return documents

    def retrieve_documents(self, query: str, top_k: int = 3) -> list:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate_response(self, query: str, context: str) -> str:
        input_text = f"Query: {query}\nContext: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=150, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run(self, query: str, top_k: int = 3) -> str:
        relevant_docs = self.retrieve_documents(query, top_k)
        combined_context = " ".join(relevant_docs)
        response = self.generate_response(query, combined_context)
        return response

# Example Usage
if __name__ == "__main__":
    # Define the folder containing PDFs
    pdf_folder = "sample_pdfs"  # Replace with the path to your folder of PDFs

    # Initialize the RAG pipeline
    rag_pipeline = RAGPipeline(pdf_folder, model_name="facebook/bart-large")

    # User query
    user_query = "What is my work experience"

    # Run the pipeline
    result = rag_pipeline.run(user_query)
    print("Generated Response:", result)
