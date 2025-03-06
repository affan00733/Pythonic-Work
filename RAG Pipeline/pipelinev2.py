import os
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ImprovedRAGPipeline:
    def __init__(self, pdf_folder: str, model_name: str = "google/flan-t5-large"):
        self.pdf_folder = pdf_folder
        self.documents = self._load_documents()
        self.vectorizer = TfidfVectorizer()
        self.doc_embeddings = self.vectorizer.fit_transform(self.documents) if self.documents else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def _load_documents(self) -> list:
        documents = []
        for file_name in os.listdir(self.pdf_folder):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, file_name)
                text = self._extract_text_from_pdf(pdf_path)
                if text.strip():
                    documents.append(text)
        if not documents:
            raise ValueError("No meaningful text could be extracted from the PDF files.")
        return documents

    def retrieve_documents(self, query: str, top_k: int = 3) -> list:
        if self.doc_embeddings is None or self.doc_embeddings.shape[0] == 0:
            raise ValueError("No document embeddings found. Ensure documents are loaded and have meaningful content.")
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate_response(self, query: str, context: str) -> str:
        if not context.strip():
            return "No relevant content found to answer the query."
        input_text = f"Query: {query}\nContext: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_beams=5,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run(self, query: str, top_k: int = 3) -> str:
        relevant_docs = self.retrieve_documents(query, top_k)
        print("Retrieved Documents for Debugging:")
        for i, doc in enumerate(relevant_docs):
            print(f"Document {i + 1}:\n{doc[:500]}...\n")  # Debugging retrieved content
        combined_context = " ".join(relevant_docs[:top_k])
        return self.generate_response(query, combined_context)


# Example Usage
if __name__ == "__main__":
    pdf_folder = "sample_pdfs"  # Replace with the folder containing PDFs
    rag_pipeline = ImprovedRAGPipeline(pdf_folder, model_name="google/flan-t5-large")

    user_query = "What is my work experience and what is my job title?"
    result = rag_pipeline.run(user_query)
    print("Generated Response:", result)
