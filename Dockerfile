FROM python:3.12-slim
#f
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/iris_model.joblib \
    PORT=8000

WORKDIR /app

COPY requirements-serving.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-serving.txt

COPY models/iris_model.joblib ./models/iris_model.joblib
COPY src ./src
RUN useradd --no-create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER 10001:10001
EXPOSE 8000

CMD ["python", "-m", "src.serve"]
