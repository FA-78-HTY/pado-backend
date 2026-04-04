FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Hugging Face Spaces exposes port 7860 by default
CMD uvicorn step2_api:app --host 0.0.0.0 --port ${PORT:-10000}
