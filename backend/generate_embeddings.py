import wikipedia, json, textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

topics = [

    # General knowledge & life skills
    "Human body", "Nutrition", "Mental health", "Exercise", "Sleep", "Stress management",
    "Time management", "Productivity", "Goal setting", "Leadership", "Public speaking",
    "Confidence building", "Communication skills", "Decision making", "Problem solving",
    "Mindfulness", "Meditation", "Habits", "Motivation", "Happiness",

    # Personal finance & business basics
    "Saving money", "Budgeting", "Personal finance", "Credit score", "Loans",
    "Investing for beginners", "Mutual funds", "Stock market basics", "Cryptocurrency basics",
    "Real estate investing", "Financial planning", "Taxes", "Insurance", "Retirement planning",
    "Entrepreneurship", "Startup basics", "Marketing", "Digital marketing",
    "Branding", "Advertising", "Sales", "Customer service", "E-commerce", "Freelancing",

    # Technology (high-level, not scientific research)
    "Artificial intelligence basics", "Machine learning basics", "Cybersecurity basics",
    "Cloud computing basics", "Computer network basics", "Operating system basics",
    "Internet of Things basics", "Smartphones", "Robotics basics", "Web development basics",
    "Mobile apps", "Social media", "Virtual reality", "Augmented reality", "3D printing",
    "Video editing", "Gaming industry", "Cryptocurrency overview", "Blockchain basics",

    # History
    "Ancient Egypt", "Indus Valley Civilization", "Mesopotamia", "Ancient Greece",
    "Roman Empire", "Ottoman Empire", "Mughal Empire", "British Empire", "French Revolution",
    "American Revolution", "Industrial Revolution", "World War I", "World War II",
    "Cold War", "Renaissance", "Egyptian Pharaohs", "Vikings", "Medieval Europe",

    # Countries & Geography
    "India", "United States", "China", "Japan", "Russia", "United Kingdom",
    "France", "Germany", "Spain", "Italy", "Australia", "Canada", "Brazil",
    "Saudi Arabia", "South Africa", "Egypt", "Pakistan", "Sri Lanka", "Nepal",
    "Tourism in India", "Famous world monuments", "Oceans of the world",
    "Mountain ranges of the world", "Rivers of the world", "Deserts of the world",
    "Islands of the world", "Himalayas", "Sahara Desert", "Amazon rainforest",

    # Civics & Society
    "Democracy", "Constitution", "Human rights", "Law and justice", "UN",
    "European Union", "NATO", "World Health Organization", "International trade",
    "Education system", "Healthcare system", "Globalization", "Immigration",

    # Literature & Art
    "Shakespeare", "Greek mythology", "Indian mythology", "Poetry", "Novels",
    "Fiction", "Non-fiction", "Painting", "Sculpture", "Photography", "Architecture",
    "History of cinema", "Animation", "Cartoons", "Graphic design",

    # Entertainment
    "Hollywood", "Bollywood", "Tollywood", "Anime", "K-pop",
    "Netflix", "YouTube", "Video games", "Esports", "Music industry",
    "Oscar awards", "Grammy awards", "Reality shows", "Sitcoms", "Action movies",

    # Sports
    "Cricket", "Football", "Basketball", "Tennis", "Badminton", "Table tennis",
    "Volleyball", "Hockey", "Athletics", "Running", "Swimming", "Cycling",
    "Chess", "Olympics", "Yoga", "Martial arts", "Gym workouts",

    # Food & Culture
    "Indian cuisine", "Chinese cuisine", "Italian cuisine", "Mexican cuisine",
    "Continental cuisine", "Vegetarian diet", "Vegan diet", "Street food",
    "Festivals of India", "Global festivals", "World religions", "Languages of the world",
    "Wedding traditions", "Traditional clothing", "Cultural diversity",

    # Travel & Lifestyle
    "Travel tips", "Solo travel", "Backpacking", "Adventure tourism",
    "Luxury travel", "Budget travel", "Work–life balance", "Minimalism",
    "Interior design", "Fashion", "Skincare", "Haircare", "Fitness",
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

corpus_text_test = Path("documents/corpus.txt").read_text(encoding="utf-8")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

chunks_text = text_splitter.split_text(corpus_text_test)

chunks = [
    {"id": i, 
     "doc_id": "wiki_dataset", 
     "chunk_id": i, 
     "text": chunk}
    for i, chunk in enumerate(chunks_text)
]

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