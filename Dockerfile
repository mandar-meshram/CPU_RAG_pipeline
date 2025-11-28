FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Install all requirements
RUN pip install --no-cache-dir -r requirements.txt

# Create the dirs
RUN mkdir -p indexes data/raw data/processed logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]