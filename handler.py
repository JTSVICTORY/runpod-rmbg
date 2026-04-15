"""
RMBG-2.0 배경 제거 - RunPod Serverless Handler

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
from torchvision import transforms

# 글로벌 모델
model = None
device = None


def load_model():
    global model, device
    if model is not None:
        return

    print("Loading RMBG-2.0 model...")
    from transformers import AutoModelForImageSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
    )
    # 모델의 dtype에 맞추기
    model = model.to(device)
    model.eval()

    print(f"RMBG-2.0 loaded on {device}, dtype={next(model.parameters()).dtype}")


def remove_background(image_b64):
    load_model()

    # base64 → PIL Image
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    original_size = image.size

    # 모델의 dtype 확인
    model_dtype = next(model.parameters()).dtype

    # 전처리
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device, dtype=model_dtype)

    # 추론
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    # 마스크
    mask = preds[0].squeeze()
    mask_np = (mask.float() * 255).byte().numpy()
    mask_img = Image.fromarray(mask_np).resize(original_size, Image.BILINEAR)

    # 알파 적용
    output = image.copy().convert("RGBA")
    output.putalpha(mask_img)

    # base64 출력
    buf = io.BytesIO()
    output.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), original_size[0], original_size[1]


def handler(event):
    try:
        inp = event.get("input", {})
        image_b64 = inp.get("image", "")

        if not image_b64:
            return {"error": "image is required"}

        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",")[1]

        result_b64, w, h = remove_background(image_b64)

        return {
            "image": result_b64,
            "format": "png",
            "width": w,
            "height": h,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
