# Financial RAG Assistant

A Retrieval-Augmented Generation (RAG) application designed to analyze financial documents (e.g., 10-K reports) and answer complex queries using LLMs.

Built with **Streamlit**, **LlamaIndex**, **Qdrant**, and **Groq (Llama 3)**.

---

## Features

* Ask natural language questions about financial reports
* Uses RAG (Retrieval-Augmented Generation) for grounded answers
* Streaming responses for better UX
* Source document citations
* Local vector database using Qdrant
* Fully free embedding model (HuggingFace BGE)


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd financial-rag
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
LLAMA_CLOUD_API_KEY=your_llama_parse_key
```

---

## Step 1: Parse PDF

```bash
python ingest.py
```

This will:

* Extract structured content from the PDF
* Save it as `parsed_output.md`

---

## Step 2: Build Vector Database

```bash
python vectorize.py
```

This will:

* Load parsed Markdown
* Chunk the document
* Generate embeddings
* Store vectors in Qdrant

---

## Step 3: Run the App

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## Planned Improvements

* Hybrid search (dense + keyword)
* Reranking for better retrieval accuracy
* Metadata filtering (company, year, section)
* Improved chunking for financial structure
* Better prompt control for numeric accuracy
* Deployment-ready architecture

---

## Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq (Llama 3.3 70B)
* **Embeddings:** HuggingFace BGE-small
* **Vector DB:** Qdrant (local)
* **Framework:** LlamaIndex
* **Parser:** LlamaParse

## License

MIT License

---
This project is a **learning + prototype system** for financial RAG.
It demonstrates core concepts but is not yet production-ready.
---
