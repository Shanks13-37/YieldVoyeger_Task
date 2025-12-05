import wikipedia, json, textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

topics = [
    "Artificial intelligence", "Machine learning", "Deep learning", "Neural network",
    "Transformer (machine learning model)", "Generative artificial intelligence",
    "Large language model", "Natural language processing", "Computer vision",
    "Reinforcement learning", "Supervised learning", "Unsupervised learning",
    "Overfitting", "Algorithm", "Data structure", "Time complexity",
    "Computational complexity theory", "Operating system", "Distributed computing",
    "Parallel computing", "Quantum computing", "Blockchain", "Cryptocurrency",
    "Cybersecurity", "Ethical hacking", "Public-key cryptography", "Cloud computing",
    "Edge computing", "Internet of Things", "Big data", "Data mining",
    "Database management system", "SQL", "NoSQL", "Computer architecture",
    "Computer network", "Transmission Control Protocol", "Internet Protocol",
    "Domain Name System", "Information security", "Software engineering",
    "Systems engineering", "Version control", "Git",

    "Space exploration", "Rocket", "Rocket propulsion", "Spaceflight", "Astronomy",
    "Astrophysics", "Astrobiology", "NASA", "ISRO", "ESA (European Space Agency)",
    "International Space Station", "Mars", "Moon", "Venus", "Jupiter",
    "Black hole", "Big Bang", "Supernova", "Milky Way", "Andromeda Galaxy",
    "Exoplanet", "Dark matter", "Dark energy", "Solar system",
    "Cosmic microwave background", "Hubble Space Telescope",
    "James Webb Space Telescope", "Gravitational wave", "Speed of light",
    "Nuclear fusion", "SpaceX", "Starship (rocket)",

    "DNA", "RNA", "Protein", "Photosynthesis", "Genetic engineering",
    "Human brain", "Neuron", "Synapse", "Stem cell", "Evolution", "Natural selection",
    "Ecology", "Enzyme", "Metabolism", "Amino acid", "Cell division",
    "Mitochondrion", "Immune system", "Homeostasis", "Blood–brain barrier", "Psychology", "Cognitive science", "Chemistry",
    "Organic chemistry", "Inorganic chemistry", "Periodic table",
    "Chemical reaction", "Particle physics", "Quantum mechanics",
    "Relativity", "Thermodynamics"

    "Quantum mechanics", "General relativity", "Special relativity", "String theory",
    "Quantum field theory", "Schrodinger equation", "Heisenberg uncertainty principle",
    "Quantum tunneling", "Electron", "Proton", "Neutron", "Newton's laws of motion",
    "Classical mechanics", "Electromagnetism", "Maxwell's equations", "Conservation of energy",
    "Conservation of momentum", "Thermodynamics", "Entropy", "Kinetic theory of gases",
    "Torque", "Angular momentum", "Simple harmonic motion", "Oscillation",
    "Projectile motion", "Fluid dynamics", "Viscosity", "Friction", "Work and energy",
    "Power (physics)", "Dark matter", "Dark energy", "Big Bang theory",
    "Gravitational wave", "Neutron star", "Supernova", "Gamma-ray burst", "Observable universe", "Multiverse", "Nuclear physics", "Nuclear fusion", "Nuclear fission",
    "Radioactivity", "Half-life (physics)", "Semiconductor"

]

dataset_text = ""

for title in topics:
    try:
        print(f"Fetching: {title}")
        summary = wikipedia.summary(title)
        dataset_text += f"{title}. {summary}\n\n"
    except Exception as e:
        print(f"Skipping {title}: {e}")

Path("documents").mkdir(exist_ok=True)
Path("documents/corpus.txt").write_text(dataset_text, encoding="utf-8")
print("\nSUCCESS — DATASET CREATED")

def simple_chunk_text(text: str, max_chars: int=500):
  paragraphs =[p.strip() for p in text.split("\n\n") if p.strip()]

  chunks =[]
  chunk_id = 0

  for p in paragraphs:
    wrapped = textwrap.wrap(p, max_chars)
    for w in wrapped:
      chunks.append({
          "id": chunk_id,
          "doc_id": "wiki_dataset",
          "chunk_id": chunk_id,
          "text" :w
      })
      chunk_id +=1
  return chunks

corpus_text_test = Path("documents/corpus.txt").read_text(encoding="utf-8")
chunks = simple_chunk_text(corpus_text_test, max_chars=500)

Path("embeddings/chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [c["text"] for c in chunks]
embeddings = embed_model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True
)

d = embeddings.shape[1]
index= faiss.IndexFlatL2(d)
index.add(embeddings.astype("float32"))
faiss.write_index(index, "embeddings/faiss.index")

print("Embeddings and FAISS index created.")