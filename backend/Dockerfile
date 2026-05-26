# Build from backend/: docker build -t rookie-trader-api .
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN mkdir -p data

ENV DATABASE_URL=sqlite:///./data/stock_data.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
