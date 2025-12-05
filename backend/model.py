import json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = "google/flan-t5-base"
tokenizer= AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

chunks = json.loads(open("embeddings/chunks.json", encoding="utf-8").read())
index = faiss.read_index("embeddings/faiss.index")

def retrieve_chunks(query, top_k =3):
  q_emb= embed_model.encode([query], convert_to_numpy=True).astype("float32")
  D, I =index.search(q_emb, top_k)
  retrieved = [chunks[i] for i in I[0]]
  return retrieved

def generate_answer(question):
  retrieved_chunks = retrieve_chunks(question)
  context = "\n\n".join([c["text"] for c in retrieved_chunks])

  prompt = f"You are a helpful AI assistant.Answer the question based ONLY on the context below.Give a detailed and complete explanation using multiple sentences. Use paragraphs, not a single long sentence.\n\nContext: \n{context}\n\nQuestion: {question}"

  input = tokenizer(prompt, return_tensors="pt")
  outputs = hf_model.generate(**input, max_new_tokens=1000)
  answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return answer