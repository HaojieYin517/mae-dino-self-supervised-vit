from PIL import Image
import numpy as np
import os
from tqdm import tqdm

dataset_path = "/workspace/hf_dataset/train"

means, stds = [], []
for fname in tqdm(os.listdir(dataset_path)[:1000]):  # sample 1000 images for estimate
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = Image.open(os.path.join(dataset_path, fname)).convert('RGB')
        arr = np.array(img) / 255.0
        means.append(arr.mean(axis=(0, 1)))
        stds.append(arr.std(axis=(0, 1)))

mean = np.array(means).mean(axis=0)
std = np.array(stds).mean(axis=0)
print("Mean:", mean)
print("Std:", std)