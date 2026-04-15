from transformers import AutoModelForImageSegmentation
print("Downloading BiRefNet model...")
model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
# float32로 변환하여 저장 (캐시에 float32로 남음)
model = model.float()
print("BiRefNet download complete, dtype:", next(model.parameters()).dtype)
