# 🎙️ DocSpeak — Intelligent Document Q&A Chatbot  

## 📌 Overview  
**DocSpeak** is an AI-powered chatbot that allows you to **ask questions about your documents** and get **accurate, conversational answers**.  
It uses **LLMs + embeddings + vector search** to understand your queries in context, and even responds with **speech output**, making the experience more natural.  

This project demonstrates how to build a **retrieval-augmented generation (RAG) system** with:  
- Context-aware document processing  
- Vector database storage  
- Interactive voice-based Q&A  

---

## 🚀 Features  
- ✅ **Context-aware Q&A** — Upload PDFs, DOCX, PPTX, HTML files and ask questions based on them  
- ✅ **LLM-powered responses** — Uses Openrouter AI + Hugging Face embeddings for precise answers  
- ✅ **Vector search with FAISS** — Efficient semantic retrieval of relevant chunks  
- ✅ **Voice interaction** — Ask via microphone, get spoken responses  
- ✅ **Conversation history** — Maintains context across questions  
- ✅ **Simple & interactive UI** — Built with Streamlit  

---

## 🛠️ Tech Stack  

### Core
- **Python 3.10+**
- **Streamlit** → Web-based interactive interface  
- **LangChain** → Document parsing, splitting, RAG pipeline  
- **FAISS** → Vector database for semantic search  
- **Hugging Face Embeddings** → High-quality text embeddings  
- **Openrouter AI API** → LLM backend for Q&A  

### Speech
- **SpeechRecognition** → Convert voice → text  
- **pyttsx3** → Convert text → speech  

### Document Processing 
- **Docling** → Enhanced parsing for all provided data files type  

---

## 📂 Project Structure 
```st
DocSpeak/
├── .vscode/                  # VSCode workspace settings
├── .gitignore                # Git ignore rules
├── all_installed.txt         # List of installed packages (for reference)
├── basic_final.py            # Main app script (Streamlit chatbot)
├── basic.py                  # Auxiliary script (starting point)
├── basic2.py                 # Auxiliary script
├── input.py                  # Basic input for LLM
├── inputf.py                 # Final input for LLM (Fireworks)
├── installed.txt             # Installed packages reference
├── llm.py                    # LLM setup
├── record.py                 # Audio recording utilities
├── sp_to_txt.py              # Speech-to-text utilities
└── speak.py                  # Text-to-speech utilities

# Generated after usage
├── data/                     # Uploaded documents and downloaded files
├── env/                      # Python virtual environment
├── recordings/               # Saved audio recordings
├── vector_db/                # FAISS vector database for embeddings
└── .env                      # Environment variables (FA_TOKEN etc.)
```

---

## ⚡ Installation  

### 1️⃣ Clone the repo  
```bash
git clone https://github.com/ashutosh0203/DocSpeak.git
cd AskAloud
```
### 2️⃣ Create virtual environment & install dependencies
```bash
python -m venv env
source env/bin/activate   # On Linux/Mac
env\Scripts\activate      # On Windows

pip install -r installed.txt
```
### 3️⃣ Setup environment variables
- Get your token from
  - Go to Openrouter AI and log in.
  - Go to Account → API Keys.
  - Click Create API Key and name it.
  - Copy the key immediately and store it safely.
  - Use it in your code or .env file:
- Create a .env file in the root:
```.env
OPENROUTER_API_KEY=your_openrouter_api_key
```
### 4️⃣ Run the app
```bash
streamlit run basic_final.py
```

---
## 🎯 Usage

### 🔹 Upload & Ask
- Upload a **PDF/DOCX/PPTX/HTML** file or **links** .
- The document is parsed, split into chunks, embedded, and stored in **FAISS vector DB**.
- Ask questions about the document → The chatbot fetches relevant chunks and generates a contextual answer.

### 🔹 Voice Interaction
- Click **🎤 Speak** → Ask a question out loud.
- The bot will transcribe → search → respond **both in text & speech**.

### 🔹 Clear Knowledge Base
- Use **🗑️ Clear Data** → Removes all stored embeddings & resets the system.

---
# 🏗️ Architecture & Workflow

## High-Level Architecture

```text
                 ┌───────────────────┐
                 │       User        │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Streamlit UI     │
                 │ (basic_final.py)  │
                 └─────────┬─────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
 ┌──────────────┐ ┌────────────────┐ ┌──────────────┐
 │ File Upload  │ │ Website Input  │ │ Voice Input  │
 │ PDF/DOCX/etc │ │ URL Content    │ │ Microphone   │
 └──────┬───────┘ └──────┬─────────┘ └──────┬───────┘
        │                │                  │
        ▼                ▼                  ▼
 ┌─────────────────────────────────────────────┐
 │        Content Extraction & Parsing         │
 └─────────────────┬───────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────┐
 │          Text Chunking & Splitting          │
 └─────────────────┬───────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────┐
 │         Embedding Generation Model          │
 └─────────────────┬───────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────┐
 │          FAISS Vector Database              │
 └─────────────────┬───────────────────────────┘
                   │
                   ▼
         User Question / Voice Query
                   │
                   ▼
 ┌─────────────────────────────────────────────┐
 │        Semantic Similarity Search           │
 └─────────────────┬───────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────┐
 │      Retrieve Relevant Document Chunks      │
 └─────────────────┬───────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────┐
 │         Openrouter AI LLM (RAG)             │
 └─────────────────┬───────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────┐
 │      Context-Aware Response Generation      │
 └───────────────┬─────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
 ┌──────────────┐  ┌──────────────┐
 │ Text Output  │  │ Speech Output│
 └──────────────┘  └──────────────┘
```

---

## Workflow

### 1. Document / Website Ingestion

* User uploads a PDF, DOCX, PPTX, HTML file, or provides a website URL.
* The content is extracted and converted into plain text.
* Large text is split into manageable chunks for efficient retrieval.

### 2. Embedding Generation

* Each chunk is converted into vector embeddings.
* Embeddings capture semantic meaning rather than simple keywords.
* Generated embeddings are stored in a FAISS vector database.

### 3. Knowledge Base Creation

* FAISS indexes all document chunks.
* The index acts as a searchable knowledge base.
* Multiple documents and websites can contribute to the same knowledge base.

### 4. User Query Processing

* User asks a question through text or voice.
* Voice input is recorded and converted into text using Speech-to-Text.

### 5. Semantic Retrieval

* The query is embedded using the same embedding model.
* FAISS performs similarity search.
* The most relevant document chunks are retrieved.

### 6. Response Generation (RAG)

* Retrieved chunks are combined with the user's question.
* The context is sent to the Openrouter AI language model.
* The model generates an accurate answer grounded in the uploaded content.

### 7. Output Delivery

* Response is displayed in the Streamlit interface.
* Optionally converted into speech using Text-to-Speech.
* User receives both textual and spoken responses.

---

## Project Components

| File             | Responsibility                                          |
| ---------------- | ------------------------------------------------------- |
| `basic_final.py` | Main Streamlit application and orchestration            |
| `llm.py`         | OpenrouterAI LLM initialization and response generation |
| `record.py`      | Audio recording functionality                           |
| `sp_to_txt.py`   | Speech-to-Text conversion                               |
| `speak.py`       | Text-to-Speech generation                               |
| `inputf.py`      | Prompt and input handling                               |
| `vector_db/`     | FAISS vector index storage                              |
| `recordings/`    | Saved audio recordings                                  |
| `data/`          | Uploaded files and downloaded web content               |

---

## Design Decisions

### Why FAISS?

* Fast similarity search
* Lightweight and local
* No external database dependency
* Scales well for document retrieval

### Why Retrieval-Augmented Generation (RAG)?

Instead of sending entire documents to the LLM:

1. Retrieve only relevant chunks.
2. Reduce token usage.
3. Improve response accuracy.
4. Keep answers grounded in uploaded content.

### Why Modular Architecture?

Each major responsibility is separated into its own module:

* Easier maintenance
* Better testability
* Simpler future upgrades
* Independent replacement of LLM, STT, or TTS providers


## 📊 Example Workflow

1. **Upload Resume**  
   Upload `resume.pdf` containing basic details about education, skills, and extracurricular activities.

2. **Ask Question**  
   Example: *"What extracurricular activities does the candidate have?"*  
   - Bot retrieves relevant sections from the resume → Summarizes → Provides an answer.

3. **Upload Updated Resume**  
   Upload `resume_updated.pdf` which now includes more extracurricular activities.

4. **Ask Follow-up Question**  
   Example: *"What extracurricular activities does the candidate have?"*  
   - Bot now retrieves from the updated knowledge base → Combines previous and new information → Provides a **more detailed answer** reflecting all activities.

5. **Add Website Content**  
   Enter a website URL (e.g., `https://example.com/article`) in the sidebar.  
   - Bot downloads the web page → Parses text content → Embeds it into the knowledge base.

6. **Ask Questions on Website Content**  
   Example: *"What are the key points of this article?"*  
   - Bot fetches relevant content from the web page → Summarizes → Answers contextually.

7. **Ask Topic-Specific Question on Website**  
   Example: *"Explain the algorithm discussed on this page."*  
   - Bot searches the web page content → Extracts algorithm details → Provides a clear explanation.

8. **Compare Across Sources**  
   Example: *"Compare this article’s algorithm with previous documents."*  
   - Bot searches all uploaded documents and web content → Combines relevant info → Provides a comprehensive, context-aware answer.


---

## 🔧 Future Improvements
- 📈 Visualization of retrieved chunks for better insight into what the AI is referencing.
- 🌍 Deploy on **Streamlit Cloud / HuggingFace Spaces** for easy access and sharing.
- 🎙️ More natural **TTS voice integration** for realistic speech output.
- 🎤 **Enhanced Voice Interaction & Interpretation**  
  - Allow the bot to handle conversational follow-ups via voice.
  - Support multiple languages and accents.
  - Improve transcription accuracy and speed for real-time Q&A.

---





