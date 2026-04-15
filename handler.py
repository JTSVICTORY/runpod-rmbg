"""
BiRefNet 배경 제거 - RunPod Serverless Handler

입력:
  - image: base64 인코딩된 이미지
  - model: "BiRefNet" (기본값)

출력:
  - image: base64 PNG (배경 제거된 투명 이미지)
  - width: 출력 이미지 너비
  - height: 출력 이미지 높이
"""
import runpod
import base64
import io
import os
import traceback

import numpy as np
import torch
from PIL import Image

# 글로벌 모델 (lazy loading)
birefnet_model = None
birefnet_transform = None


def load_model():
    """BiRefNet 모델 로드"""
    global birefnet_model, birefnet_transform

    if birefnet_model is not None:
        return

    print("Loading BiRefNet model...")
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True
    ).to(device).eval()

    birefnet_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"BiRefNet loaded on {device}")


def remove_background(image_b64):
    """배경 제거 실행"""
    load_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # base64 → PIL Image
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    original_size = image.size

    # 전처리
    input_tensor = birefnet_transform(image).unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        preds = birefnet_model(input_tensor)[-1].sigmoid().cpu()

    # 마스크 생성
    mask = preds[0].squeeze()
    mask = (mask * 255).byte().numpy()
    mask_image = Image.fromarray(mask).resize(original_size, Image.BILINEAR)

    # 알파 채널 적용
    result = image.copy().convert("RGBA")
    result.putalpha(mask_image)

    # PIL → base64
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
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
