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
from torchvision import transforms

# 글로벌 모델
model = None
device = None
model_dtype = None


def load_model():
    global model, device, model_dtype
    if model is not None:
        return

    print("Loading BiRefNet model...")
    from transformers import AutoModelForImageSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # float32로 로드 후 GPU로 이동
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True,
    )

    # 모델 전체를 float32로 강제 변환 후 GPU로
    model = model.float().to(device).eval()
    model_dtype = torch.float32

    # 확인
    actual_dtype = next(model.parameters()).dtype
    print(f"BiRefNet loaded on {device}, dtype={actual_dtype}")

    # 만약 여전히 half이면 입력을 half로 맞추기
    if actual_dtype != torch.float32:
        model_dtype = actual_dtype
        print(f"Model is {actual_dtype}, will match input dtype")


def remove_background(image_b64):
    load_model()

    # base64 → PIL Image
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    original_size = image.size

    # 전처리
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device).to(model_dtype)

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
