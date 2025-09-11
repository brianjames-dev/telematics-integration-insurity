# Small, fast, works well with sklearn/pandas
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

# Add source
COPY . .
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn","src.serve.api:app","--host","0.0.0.0","--port","8000","--workers","1"]
