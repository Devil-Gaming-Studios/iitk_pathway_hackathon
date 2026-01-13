FROM pathwaycom/pathway:latest

# 🔴 REQUIRED for Pathway inside Docker
ENV PATHWAY_SPAWN_ARGS="--runner threaded"

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your app
COPY . /app

# Run your pipeline
CMD ["python", "app.py"]
