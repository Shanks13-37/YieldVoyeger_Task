# RAG Search Engine (FAISS + SentenceTransformer + FLAN-T5)

A lightweight Retrieval-Augmented Generation (RAG) system built using:
- **Sentence-Transformers** for text embeddings  
- **FAISS** for fast vector search  
- **FLAN-T5** for answer generation  
- **Flask** backend  
- **HTML + CSS** frontend 

## 1] Dataset creation:
For dataset I generated my own dataset directly from wikipedia to get more control over the topics.
I used python to generate wikipedia summaries and stored it in corpus.txt.

## 2] Chunking the dataset:
Large text is difficult for embedding models and LLMs to process directly.
So I split the dataset into smaller pieces ("chunks") of ~500 characters and chunk overlap ~50, which is perfect for FAISS.
Each chunk contains:
-id
-doc_id
-chunk_id
-the text itself

## 3]Embeddings:
To enable similarity search, each chunk must be converted into a vector representation.
I used: all-MiniLM-L6-v2(Sentence transformer) as it is fast, lightweight, free to use, high accuracy for FAISS and embedding dimension =384.
This results in an embedding matrix(numpy array) of shape (num_chunks, 384)

## 4] Vector Search using FAISS:
FAISS(Facebook AI Similarity Search) is the engine that performs similarity search over the chunk embeddings.
Type: IndexFlatL2
Steps:
-Initialize a FAISS index using L2 distance
-Add all embeddings to the index
-Use it to retrieve top-k relevant chunks for a user query

## 5]Retrieval function:
I implemented retrieve_chunks(query, top_k=5) which:
-Converts the user question into an embedding
-Searches FAISS
-Returns the top k most relevant chunks
These retrieved chunks become the “context” fed into the LLM.
This is the Retrieval part of RAG.

## 6]Answer Generation(LLM with context):
Tp generate the final answer, i constructed a prompt which forces the model to not halluciante, rely only on retrieved data and provice a grounded answer
I used googles- google/flan-t5-base model

## 7] UI:
Created a simple UIusing html css and connected using flask as backend

# END RESULT:
A fully functional Mini RAG Search Engine that can answer domain-specific questions using retrieved evidence from a custom-built Wikipedia dataset complete with an interactive UI.
