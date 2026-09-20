from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
# 1. Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# String into vector database
FILE_PATH = "ir.docx"
print(f"Loading document from {FILE_PATH}...")
loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    # separators=['\n\n', '\n', '.', ','],
    chunk_size=1000,  # max tokens/characters per chunk
    chunk_overlap=50 # overlap for context preservation

)
print("text split started");
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# 4. Store chunks in FAISS vector DB
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Save the index
vectorstore.save_local("faiss_doc_index")

print(f"Stored {len(chunks)} chunks for semantic search.")


# 3. Load FAISS index (allow_dangerous_deserialization=True is required)
vectorstore = FAISS.load_local(
    "faiss_doc_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. Hugging Face inference endpoint (free model)
# You can replace with any free HF model ID

client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ["HF_TOKEN"],  # Make sure HF_TOKEN is set in env
)

def ask_document(query):
    # Retrieve context from FAISS
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    # Combine query with retrieved context for the LLM
    prompt = f"Use the following document context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # You can change to another fireworks-supported model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    return completion.choices[0].message["content"]

# Example usage:
print(ask_document("What is SHAP?"))