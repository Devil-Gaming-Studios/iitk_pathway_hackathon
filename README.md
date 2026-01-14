# iitk_pathway_hackathon
🔥 Misinformation Risk Dashboard
Project Overview

The Misinformation Risk Dashboard is a real-time web application designed to detect and visualize trending fake news, sensational headlines, and misinformation across multiple sources. Using state-of-the-art NLP models, the dashboard classifies articles based on credibility and assigns a risk score, helping users quickly identify potentially misleading or sensational news.

Key Features

Real-time detection of trending news events

Automatic risk scoring for misinformation using zero-shot classification

Emotion classification of articles (fear, anger, joy, sadness, trust)

Visualizations for:

Article trends over time

Event velocity (number of articles per event)

Emotion distribution per event

Circular risk indicators

Auto-refreshing interface for live updates

User-friendly dashboards with detailed article views

Application Architecture
1. Data Ingestion

Uses a streaming CSV (stream.csv) to simulate continuous news ingestion

Processes articles in near real time using the Pathway framework

Computes semantic embeddings using SentenceTransformer

2. Processing & Risk Analysis

Groups semantically similar articles using cosine similarity

Assigns event IDs for clustering

Uses Hugging Face zero-shot classification to analyze:

Credibility and sensationalism

Emotional tone of articles

Computes a weighted misinformation risk score

3. Visualization (Streamlit Frontend)

Home Tab: Trending events, emotions, and risk indicators

News Trend Analytics Tab: Event velocity and historical trends

Fully interactive, auto-refreshing dashboard

4. Outputs

articles.jsonl – Processed article data

velocity.jsonl – Event-wise article velocity data

Project Setup

Follow the steps below to run the project locally.

Step 1: Clone the Repository
git clone https://github.com/your-username/misinformation-dashboard.git
cd misinformation-dashboard

Step 2: Open Docker Desktop

Make sure Docker Desktop is installed and running before proceeding.

Step 3: Build the Docker Image

From the project root directory, run:

docker build --no-cache -t dataquest-pathway .


This builds the Docker image required to run the Pathway streaming pipeline.

Step 4: Run the Pathway Pipeline

Run the following command to start the streaming pipeline:

docker run -it --rm \
  -v ${PWD}:/app \
  dataquest-pathway \
  python /app/app.py


What this does:

Mounts the current project directory into the container

Executes app.py

Continuously processes streaming data

Generates:

output/articles.jsonl

output/velocity.jsonl

⚠️ Keep this process running while using the dashboard.

Step 5: Install Python Dependencies (Local)

Open a new terminal (outside Docker) and run:

pip install -r requirements.txt

Step 6: Run the Streamlit Dashboard
streamlit run dashboard.py


Open the URL shown in the terminal (usually http://localhost:8501)

The dashboard will auto-refresh and display live analytics

Data Flow Summary

stream.csv simulates incoming news data

app.py processes articles and computes trends

Output files are continuously updated

dashboard.py visualizes real-time results

Notes

No API keys required

Fully open-source models

Runs offline after setup

Docker ensures reproducible execution
