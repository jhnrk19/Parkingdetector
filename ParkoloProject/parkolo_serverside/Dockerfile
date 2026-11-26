# Parkolóhely-foglaltság detektáló szerver – processzoros változat (videókártya nélküli géphez; lassú, próbához való).

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core ca-certificates curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/models \
 && curl -fsSL -o /opt/models/yolov8x-obb.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-obb.pt

COPY requirements.txt /app/requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install "ultralytics>=8.3.0" -r /app/requirements.txt

WORKDIR /app
COPY server.py vision_pipeline.py /app/

ENV YOLO_MODEL_PATH=/opt/models/yolov8x-obb.pt \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_AUTOINSTALL=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    PK_DATA=/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PK_PORT', '8000'), timeout=4)"

CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PK_PORT:-8000} --log-level warning"]
