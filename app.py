from datetime import datetime, timedelta, timezone
import numpy as np
import pathway as pw
from sentence_transformers import SentenceTransformer
import logging

# ------------------------ Logging ------------------------
# Set up logging to track pipeline progress and info messages
logging.basicConfig(level=logging.INFO)

# -------------------- Load Embedding Model --------------------
# Using Sentence-BERT for fast semantic embeddings of article text
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- Read CSV as Streaming Table --------------------
# Define schema from CSV file (infers column names and types)
schema = pw.schema_from_csv("data/stream.csv")

# Use Pathway's demo replay_csv function to simulate continuous streaming of news
# 'input_rate' controls how fast rows are read (0.05 = slower replay)
data = pw.demo.replay_csv(path='data/stream.csv', schema=schema, input_rate=0.1)

# -------------------- Column Normalization --------------------
# No additional normalization needed; 'text' column already contains full article content

# -------------------- User Defined Functions (UDFs) --------------------
@pw.udf
def embed(text: str) -> list:
    """
    Convert article text into a dense embedding vector using Sentence-BERT.
    Returns a list of floats representing semantic features.
    """
    return embedder.encode(text).tolist()

@pw.udf
def cosine_similarity(a: list, b: list) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns 0 if either vector is zero.
    """
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

# -------------------- Compute Embeddings --------------------
# Add embedding column to each article row for semantic similarity calculations
data = data.with_columns(
    embedding=embed(data.text)
)

# -------------------- Event ID Assignment (Stateful) --------------------
SIM_THRESHOLD = 0.78  # Similarity threshold to assign to existing event
event_centroids = {}   # Stores centroid embeddings of each event

@pw.udf
def assign_event_id(embedding: list) -> str:
    """
    Assign each article to an event based on similarity to existing centroids.
    If similarity >= SIM_THRESHOLD, joins the event; otherwise, creates new event.
    """
    best_event = None
    best_score = 0.0

    # Compare with existing event centroids
    for eid, centroid in event_centroids.items():
        e = np.array(embedding)
        c = np.array(centroid)
        # Cosine similarity
        score = float(np.dot(e, c) / (np.linalg.norm(e) * np.linalg.norm(c) + 1e-9))
        if score > best_score:
            best_score = score
            best_event = eid

    # Assign to best matching event or create a new one
    if best_score >= SIM_THRESHOLD:
        return best_event
    else:
        new_id = f"event_{len(event_centroids) + 1}"
        event_centroids[new_id] = embedding
        return new_id

# Apply event assignment to the data
data = data.with_columns(
    event_id=assign_event_id(data.embedding)
)

# -------------------- Parse Timestamps --------------------
# Convert timestamp strings to datetime objects for temporal aggregation
fmt = "%Y-%m-%dT%H:%M:%S"
data = data.with_columns(timestamp = pw.this.timestamp.dt.strptime(fmt))

# -------------------- Compute Event Velocity --------------------
# Tumbling windows of 30 minutes to calculate number of articles per event
velocity = data.windowby(
    data.timestamp,
    window=pw.temporal.tumbling(duration=timedelta(minutes=120))
).groupby(
    pw.this.event_id, 
    pw.this._pw_window_start
).reduce(
    pw.this.event_id,
    pw.this._pw_window_start,
    article_count=pw.reducers.count(),
)

logging.info("Computing GLOBAL_AVG from data...")  # Placeholder log

# -------------------- Write Outputs --------------------
# Write enriched article data to JSONL for dashboard consumption
pw.io.jsonlines.write(
    data,
    "output/articles.jsonl",
)

# Write velocity data (articles per event per window) to JSONL
pw.io.jsonlines.write(
    velocity,
    "output/velocity.jsonl",
)

# -------------------- Run Pipeline --------------------
# Starts the Pathway streaming engine in continuous mode
pw.run()
# ------------------------ End of Pipeline ------------------------
