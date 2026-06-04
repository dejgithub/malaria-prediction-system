FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Debug: check architecture and network
RUN python --version && pip --version && uname -m \
 && curl -sI https://pypi.org | head -1

COPY requirements.txt .
ENV PIP_INDEX_URL=https://pypi.org/simple/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p model reports datasets

EXPOSE 8000

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

CMD gunicorn backend.app:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 1 --timeout 300
