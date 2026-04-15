"""
BiRefNet 배경 제거 - RunPod Serverless Handler

입력:
  - image: base64 인코딩된 이미지

출력:
  - image: base64 PNG (배경 제거된 투명 이미지)
  - width: 출력 이미지 너비
  - height: 출력 이미지 높이
"""
import runpod
import base64
import io
import traceback

import numpy as np
import torch
from PIL import Image

# 글로벌 모델 (lazy loading)
birefnet_pipeline = None


def load_model():
    """BiRefNet 파이프라인 로드 (transformers pipeline 방식)"""
    global birefnet_pipeline

    if birefnet_pipeline is not None:
        return

    print("Loading BiRefNet pipeline...")
    from transformers import pipeline

    birefnet_pipeline = pipeline(
        "image-segmentation",
        model="ZhengPeng7/BiRefNet",
        trust_remote_code=True,
        device=0 if torch.cuda.is_available() else -1,
    )

    print("BiRefNet pipeline loaded")


def remove_background(image_b64):
    """배경 제거 실행"""
    load_model()

    # base64 → PIL Image
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    original_size = image.size

    # 파이프라인 실행
    result = birefnet_pipeline(image)

    # 결과에서 마스크 추출
    mask = result[0]["mask"] if isinstance(result, list) else result["mask"]

    # 마스크를 원본 크기로 리사이즈
    if mask.size != original_size:
        mask = mask.resize(original_size, Image.BILINEAR)

    # 알파 채널 적용
    output = image.copy().convert("RGBA")
    output.putalpha(mask)

    # PIL → base64
    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return result_b64, original_size[0], original_size[1]


def handler(event):
    """RunPod handler"""
    try:
        input_data = event.get("input", {})
        image_b64 = input_data.get("image", "")

        if not image_b64:
            return {"error": "image is required"}

        # data URL prefix 제거
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",")[1]

        result_b64, width, height = remove_background(image_b64)

        return {
            "image": result_b64,
            "format": "png",
            "width": width,
            "height": height,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
