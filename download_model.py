from transformers import AutoModelForImageSegmentation
print("Downloading RMBG-2.0 model...")
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
print("RMBG-2.0 download complete")
