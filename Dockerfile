FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir runpod pillow numpy transformers torchvision huggingface_hub einops kornia timm

# BiRefNet 모델 + 커스텀 코드를 빌드 시점에 완전히 다운로드
RUN python -c "from transformers import AutoModelForImageSegmentation; m = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True); print('BiRefNet downloaded, dtype:', next(m.parameters()).dtype)"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
