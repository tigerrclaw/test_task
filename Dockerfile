FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOTWORK_IN_CONTAINER=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY eval ./eval
COPY data ./data
COPY pytest.ini .
COPY run_tests.sh .
RUN chmod +x /app/run_tests.sh

ENTRYPOINT ["python", "-m", "hotwork"]
