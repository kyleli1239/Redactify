FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8081

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pdf_service.py ai_service.py redaction_knowledge.py feedback_store.py ./
COPY training_data ./training_data
RUN mkdir -p /app/data

EXPOSE 8081

CMD ["python", "app.py"]
