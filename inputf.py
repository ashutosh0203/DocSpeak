import os
from fireworks import LLM
from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file  
FA_TOKEN = os.getenv("FA_TOKEN")
# Set up LLM with Fireworks API

llm = LLM(model="accounts/fireworks/models/gpt-oss-20b", api_key=FA_TOKEN ,deployment_type="serverless")

# Document loading and indexing (your FAISS logic)

# file path
FILE_PATH = "https://www.geeksforgeeks.org/dsa/rabin-karp-algorithm-for-pattern-searching/"
# FILE_PATH = "ir.docx"
# load document
print(f"Loading document from {FILE_PATH}...")
loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()
#  splitting text into chunks
print("text split started")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = splitter.split_documents(docs)
# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# create vectorstore
print("Creating vectorstore...")
vectorstore = FAISS.from_documents(chunks, embeddings)
# Save the vectorstore
vectorstore.save_local("faiss_doc_index")

# Load the vectorstore
vectorstore = FAISS.load_local("faiss_doc_index", embeddings, allow_dangerous_deserialization=True)
# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def ask_document(query):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.2
    )
    return response.choices[0].message.content

print(ask_document("how does the Rabin-Karp algorithm work?"))
