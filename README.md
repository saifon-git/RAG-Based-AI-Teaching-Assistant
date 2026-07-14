# RAG-Based-AI-Teaching-Assistant
A Retrieval-Augmented Generation (RAG) based AI teaching assistant for the Sigma Web Development Course. It uses Whisper for transcription, custom chunking, BGE-M3 embeddings, cosine similarity for retrieval, and Llama 3.2 via Ollama to deliver accurate, context-aware answers with relevant video timestamps.

<img width="1920" height="1080" alt="Screenshot (673)" src="https://github.com/user-attachments/assets/6a62ba70-cdea-4732-8a43-e0362e6a62b3" />
<img width="1920" height="1080" alt="Screenshot (672)" src="https://github.com/user-attachments/assets/5b43c053-340a-4c5f-85cf-f32cde28645b" />


# Project Workflow

This document explains the end-to-end workflow of the **RAG-Based AI Teaching Assistant** project, from processing lecture videos to generating intelligent answers.

---

# Workflow Overview

```
Lecture Video
      │
      ▼
Video Preprocessing (FFmpeg)
      │
      ▼
Speech-to-Text (Whisper)
      │
      ▼
Transcript Generation
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation (Ollama - bge-m3)
      │
      ▼
Store Embeddings (Joblib)
      │
      ▼
User Question
      │
      ▼
Question Embedding
      │
      ▼
Cosine Similarity Search
      │
      ▼
Top Relevant Chunks
      │
      ▼
LLM (RAG)
      │
      ▼
Generated Answer
```

---

# Detailed Workflow

## Step 1: Video Input

- The system accepts recorded lecture videos.
- Supported video formats include MP4, AVI, MKV, and MOV.

Output:
- Lecture video ready for processing.

---

## Step 2: Video Preprocessing

The video is converted into an audio file using **FFmpeg**.

Purpose:
- Remove unnecessary video frames.
- Extract clear audio for transcription.

Output:
- MP3 audio file.

---

## Step 3: Speech-to-Text

The extracted audio is passed to **OpenAI Whisper**.

Purpose:
- Convert spoken lecture into text.
- Preserve timestamps.

Output:
- Transcript with timestamps.

---

## Step 4: Text Chunking

The transcript is divided into smaller chunks.

Each chunk contains:
- Chunk number
- Start timestamp
- End timestamp
- Lecture title
- Chunk text

Purpose:
- Improve retrieval accuracy.
- Fit within the LLM context window.

Output:
- JSON chunks.

---

## Step 5: Embedding Generation

Each chunk is converted into a vector embedding using the **bge-m3** embedding model running on **Ollama**.

Purpose:
- Represent semantic meaning mathematically.

Output:
- Embedding vectors.

---

## Step 6: Store Embeddings

Generated embeddings are stored using **Joblib**.

Purpose:
- Avoid regenerating embeddings every time.
- Faster retrieval.

Output:
- `embeddings.joblib`

---

## Step 7: User Query

The user enters a question.

Example:

> What is machine learning?

---

## Step 8: Query Embedding

The user's question is converted into an embedding using the same embedding model.

Purpose:
- Ensure both query and document vectors exist in the same vector space.

---

## Step 9: Similarity Search

Cosine Similarity compares the query embedding with every stored chunk embedding.

Purpose:
- Retrieve the most relevant lecture chunks.

Output:
- Top matching chunks.

---

## Step 10: Retrieval-Augmented Generation (RAG)

The retrieved chunks are supplied as context to the Large Language Model.

Purpose:
- Prevent hallucinations.
- Generate answers grounded in lecture content.

Output:
- Accurate context-aware response.

---

# Technologies Used

- Python
- FFmpeg
- Whisper
- Ollama
- bge-m3 Embedding Model
- NumPy
- Pandas
- Joblib
- Scikit-learn (Cosine Similarity)

---

# Project Architecture

```
Video
  │
  ▼
FFmpeg
  │
  ▼
Whisper
  │
  ▼
Transcript
  │
  ▼
Chunking
  │
  ▼
Embeddings
  │
  ▼
Vector Storage
  │
  ▼
User Query
  │
  ▼
Query Embedding
  │
  ▼
Similarity Search
  │
  ▼
Relevant Chunks
  │
  ▼
LLM
  │
  ▼
Final Answer
```

---

# Advantages

- Accurate answers using lecture context.
- Reduced hallucinations.
- Fast retrieval through vector search.
- Supports multilingual lecture transcription.
- Easily extendable with new lecture videos.

---

# Future Improvements

- Web-based interface.
- Multi-user support.
- Vector database integration (FAISS, ChromaDB, Pinecone).
- GPU acceleration.
- Support for PDF and document uploads.
- Streaming responses.
