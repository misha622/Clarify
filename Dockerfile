# Clarify Autonomous Security Layer
#
# docker build -t clarify .
# docker run -p 8000:8000 -v ./config:/app/config -v ./models:/app/models clarify

FROM python:3.12-slim

LABEL org.opencontainers.image.title="Clarify"
LABEL org.opencontainers.image.description="Autonomous Security Layer with explainable ML"
LABEL org.opencontainers.image.version="0.1.0"

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY src/ ./src/
COPY config/ ./config/

# Создаём папку для моделей (монтируется через volume или копируется)
RUN mkdir -p /app/models

# Порт для API
EXPOSE 8000

# По умолчанию — интерактивное демо
CMD ["python", "-m", "src.ui.alert_card"]