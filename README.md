# 🔥 Misinformation Risk Dashboard
### *IITK Pathway Hackathon*

[![Pathway](https://img.shields.io/badge/Framework-Pathway-brightgreen)](https://pathway.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/NLP-HuggingFace-yellow)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-blue)](https://www.docker.com/)

The **Misinformation Risk Dashboard** is a real-time web application designed to detect and visualize trending fake news, sensational headlines, and misinformation across multiple sources. Using state-of-the-art NLP models, it assigns a risk score to help users identify misleading content instantly.

---

## 🎯 Project Overview
In a world of rapid information flow, identifying "fake news" is a race against time. This project leverages the **Pathway framework** for high-performance stream processing to:
* **Detect** trending news events as they happen.
* **Analyze** emotional triggers (Fear, Anger, Trust).
* **Score** articles based on credibility and sensationalism.
* **Visualize** event velocity and risk distributions.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Real-time Detection** | Instant grouping of trending news events using semantic embeddings. |
| **Risk Scoring** | Automatic misinformation scoring using zero-shot classification. |
| **Emotion Analysis** | Classifies articles into Fear, Anger, Joy, Sadness, and Trust. |
| **Dynamic Analytics** | Visualizes event velocity (growth rate) and article trends over time. |
| **Auto-Refresh UI** | A live-updating Streamlit interface for hands-free monitoring. |

---

## 🏗️ Application Architecture

### 1. Data Ingestion 📥
* Uses a streaming CSV (`stream.csv`) to simulate continuous news ingestion.
* Processes data in near real-time via **Pathway**.
* Generates semantic embeddings using `SentenceTransformer`.

### 2. Processing & Risk Analysis 🧠
* **Clustering:** Groups similar articles using cosine similarity.
* **Classification:** Uses Hugging Face zero-shot models for Credibility and Emotion analysis.
* **Risk Engine:** Computes a weighted risk score based on model outputs.

### 3. Visualization & Output 📊
* **Home Tab:** Circular risk indicators and emotion distribution.
* **News Trend Analytics:** Deep dive into event velocity and history.
* **Data Exports:** Continuous updates to `articles.jsonl` and `velocity.jsonl`.

---

## 🛠️ Project Setup

### Step 1: Clone the Repository
```bash
git clone [https://github.com/your-username/misinformation-dashboard.git](https://github.com/your-username/misinformation-dashboard.git)
cd misinformation-dashboard

Step 2: Build the Image (Docker)
Bash

docker build --no-cache -t dataquest-pathway .
Step 3: Launch the Pathway Pipeline
Run the container to start processing the live stream:

Bash

docker run -it --rm -v ${PWD}:/app dataquest-pathway python /app/app.py
⚠️ Important: Keep this terminal running to ensure the dashboard has live data to read.

Step 4: Run the Dashboard
Open a new terminal and run the Streamlit UI:

Bash

pip install -r requirements.txt
streamlit run dashboard.py
