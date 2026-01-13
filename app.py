from datetime import datetime, timedelta,timezone

import numpy as np
import pathway as pw
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
# --------------------------------------------------
# Load embedding model
# --------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# Read CSV as streaming table (continuous replay)
# --------------------------------------------------
schema = pw.schema_from_csv("data/stream.csv")
# Use replay_csv with loop enabled to continuously read data
data = pw.demo.replay_csv(path='data/stream.csv', schema=schema, input_rate=0.05)

# --------------------------------------------------
# Normalize columns (keep full text as-is)
# --------------------------------------------------
# data already has the 'text' column from stream.csv with full descriptions

# --------------------------------------------------
# UDFs
# --------------------------------------------------
@pw.udf
def embed(text: str) -> list:
    return embedder.encode(text).tolist()

@pw.udf
def cosine_similarity(a: list, b: list) -> float:
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

# --------------------------------------------------
# Compute embeddings
# --------------------------------------------------
data = data.with_columns(
    embedding=embed(data.text)
)

# --------------------------------------------------
# EVENT ID ASSIGNMENT (STATEFUL)
# --------------------------------------------------
SIM_THRESHOLD = 0.78
event_centroids = {}

@pw.udf
def assign_event_id(embedding: list) -> str:
    import numpy as np

    best_event = None
    best_score = 0.0

    for eid, centroid in event_centroids.items():
        e = np.array(embedding)
        c = np.array(centroid)
        score = float(np.dot(e, c) / (np.linalg.norm(e) * np.linalg.norm(c) + 1e-9))
        if score > best_score:
            best_score = score
            best_event = eid

    if best_score >= SIM_THRESHOLD:
        return best_event
    else:
        new_id = f"event_{len(event_centroids) + 1}"
        event_centroids[new_id] = embedding
        return new_id

data = data.with_columns(
    event_id=assign_event_id(data.embedding)
)
# --------------------------------------------------
# PARSE TIMESTAMPS
fmt = "%Y-%m-%dT%H:%M:%S"
data = data.with_columns(timestamp = pw.this.timestamp.dt.strptime(fmt))
# --------------------------------------------------
# VELOCITY (articles per event)
# --------------------------------------------------

velocity = data.windowby(
    data.timestamp,
    window=pw.temporal.tumbling(duration=timedelta(minutes=30))
).groupby(pw.this.event_id, pw.this._pw_window_start).reduce(
    pw.this.event_id,
    pw.this._pw_window_start,
    article_count=pw.reducers.count(),
)

logging.info("Computing GLOBAL_AVG from data...")
# --------------------------------------------------
# OUTPUTS
# --------------------------------------------------
pw.io.jsonlines.write(
    data,
    "output/articles.jsonl",
)

pw.io.jsonlines.write(
    velocity,
    "output/velocity.jsonl",
)

# --------------------------------------------------
# RUN PIPELINE (CONTINUOUS MODE)
# --------------------------------------------------
pw.run()
