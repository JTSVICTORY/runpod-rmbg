FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir runpod pillow numpy transformers torchvision huggingface_hub

COPY download_model.py .
RUN python download_model.py

COPY handler.py .

CMD ["python", "-u", "handler.py"]
