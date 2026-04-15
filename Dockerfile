FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Python 의존성
RUN pip install --no-cache-dir \
    runpod \
    pillow \
    numpy \
    transformers \
    torch \
    torchvision \
    huggingface_hub

# BiRefNet 모델 사전 다운로드 (~900MB)
RUN python -c "
from huggingface_hub import hf_hub_download
import os

# BiRefNet-HR 모델 다운로드
model_id = 'ZhengPeng7/BiRefNet'
os.makedirs('/app/models', exist_ok=True)
for f in ['model.safetensors', 'config.json', 'preprocessor_config.json']:
    try:
        hf_hub_download(repo_id=model_id, filename=f, local_dir='/app/models/BiRefNet')
        print(f'Downloaded {f}')
    except:
        print(f'Skipped {f}')
print('BiRefNet model downloaded')
"

# 핸들러 복사
COPY handler.py .

# RunPod Serverless 실행
CMD ["python", "-u", "handler.py"]
