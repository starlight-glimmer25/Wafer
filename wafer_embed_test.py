# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model = model.to(device)
model.eval()
print("DINOv2 ViT-S/14 loaded")

data_path = "/users/ysong25/wafer_project/data/test_unsupervised.pkl"

df = pd.read_pickle(data_path)
print("Dataset loaded:", df.shape)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def preprocess_wafer(wafer):
    img = np.array(wafer).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)
    return tensor

all_embeddings = []
failed_indices = []

for idx in tqdm(range(len(df)), desc="Embedding VAL wafers"):

    wafer = df.iloc[idx]["waferMap"]
    try:
        input_tensor = preprocess_wafer(wafer)
        with torch.no_grad():
            emb = model(input_tensor)
        emb = emb.squeeze().cpu().numpy()
        all_embeddings.append(emb)

    except Exception as e:
        print(f"Failed at index {idx}: {str(e)}")
        all_embeddings.append(None)
        failed_indices.append(idx)

save_dir = "/users/ysong25/wafer_project/embeddings_test"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "test_embeddings_dino_vits14.npy"),
        np.array(all_embeddings, dtype=object))
np.save(os.path.join(save_dir, "test_failed_indices.npy"),
        np.array(failed_indices))

print("======================================")
print("EMBEDDINGS FINISHED")
print("Total:", len(df))
print("Failed:", len(failed_indices))
print("Saved to:", save_dir)
print("======================================")
