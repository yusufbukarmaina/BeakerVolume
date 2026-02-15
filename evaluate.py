import re
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration
from config import Config
from data import load_dataset_split, detect_image_col, split_dataset, prepare

cfg = Config()

VOL_RE = re.compile(r"(\d+(?:\.\d+)?)")

def parse(txt):
    m = VOL_RE.search(txt)
    return float(m.group(1)) if m else None

def metrics(y, p):
    pairs = [(a,b) for a,b in zip(y,p) if b is not None]
    yt = np.array([a for a,_ in pairs])
    yp = np.array([b for _,b in pairs])
    return {
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": mean_squared_error(yt, yp, squared=False),
        "R2": r2_score(yt, yp)
    }

print("Run after training is complete.")
