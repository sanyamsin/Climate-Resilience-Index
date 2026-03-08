FROM python:3.11-slim

LABEL maintainer="Tresor Niyomwungere"
LABEL description="AdaptAction Climate Resilience Index Dashboard"
LABEL version="1.0.0"

# System dependencies for GeoPandas / GDAL
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY climate_resilience/ ./climate_resilience/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY data/ ./data/

ENV PYTHONPATH=/app
ENV DASH_DEBUG=false

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8050/ || exit 1

CMD ["python", "dashboard/app.py"]
