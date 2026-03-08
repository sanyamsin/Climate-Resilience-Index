FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir numpy pandas geopandas plotly dash dash-bootstrap-components scipy

COPY climate_resilience/ ./climate_resilience/
COPY dashboard/ ./dashboard/

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "dashboard/app.py"]
