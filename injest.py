import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

def load_data():
    races = pd.read_csv(f"{DATA_DIR}/races.csv")
    results = pd.read_csv(f"{DATA_DIR}/results.csv")
    drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
    constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
    return races, results, drivers, constructors

def build_sentences(races, results, drivers, constructors):
    merged = results.merge(races, on="raceId") \
                    .merge(drivers, on="driverId") \
                    .merge(constructors, on="constructorId")
    sentences = []
    for _, row in merged.iterrows():
        try:
            name = f"{row['forename']} {row['surname']}"
            team = row['name_y'] if 'name_y' in row else row.get('name', 'Unknown')
            race = row['name_x'] if 'name_x' in row else 'Unknown Race'
            year = row.get('year', '?')
            pos = row.get('positionText', '?')
            grid = row.get('grid', '?')
            laps = row.get('laps', '?')
            fastest = row.get('fastestLapTime', '')
            text = f"In {year} {race}, {name} ({team}) started P{grid}, finished P{pos}, completed {laps} laps."
            if fastest and str(fastest) != 'nan':
                text += f" Fastest lap: {fastest}."
            sentences.append(text)
        except Exception:
            continue
    return sentences

def main():
    print("Loading CSV data...")
    races, results, drivers, constructors = load_data()
    print("Building sentences...")
    sentences = build_sentences(races, results, drivers, constructors)
    print(f"Total records: {len(sentences)}")
    print("Loading embedding model (first time may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding and storing in ChromaDB...")
    batch_size = 500
    db = None
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        if db is None:
            db = Chroma.from_texts(batch, embeddings, persist_directory=CHROMA_DIR)
        else:
            db.add_texts(batch)
        print(f"  Processed {min(i+batch_size, len(sentences))}/{len(sentences)}")
    db.persist()
    print("Done! ChromaDB saved to chroma_db/")

if __name__ == "__main__":
    main()