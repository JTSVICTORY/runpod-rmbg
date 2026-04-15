from huggingface_hub import hf_hub_download
import os

model_id = "ZhengPeng7/BiRefNet"
os.makedirs("/app/models/BiRefNet", exist_ok=True)
for f in ["model.safetensors", "config.json", "preprocessor_config.json"]:
    try:
        hf_hub_download(repo_id=model_id, filename=f, local_dir="/app/models/BiRefNet")
        print("Downloaded " + f)
    except Exception as e:
        print("Skipped " + f + ": " + str(e))
print("BiRefNet model download complete")
