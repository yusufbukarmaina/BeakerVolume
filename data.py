import re
import random
from datasets import load_dataset
from datasets.features import Image as HFImage

VOL_RE = re.compile(r"_(?:v)?(\d+(?:\.\d+)?)\s*(?:mL|ml)_", re.IGNORECASE)

def extract_volume(name):
    m = VOL_RE.search(name)
    if not m:
        return None
    return float(m.group(1))

def load_dataset_split(cfg):
    dsd = load_dataset(cfg.HF_DATASET_ID)
    ds = dsd["train"] if "train" in dsd else dsd[list(dsd.keys())[0]]
    return ds

def detect_image_col(ds):
    for col, feature in ds.features.items():
        if isinstance(feature, HFImage):
            return col
    raise ValueError("No image column detected.")

def split_dataset(ds, cfg):
    n = len(ds)
    idx = list(range(n))
    random.seed(cfg.SEED)
    random.shuffle(idx)

    n_train = int(n * cfg.TRAIN_FRAC)
    n_val = int(n * cfg.VAL_FRAC)

    train = ds.select(idx[:n_train])
    val = ds.select(idx[n_train:n_train+n_val])
    test = ds.select(idx[n_train+n_val:])

    return train, val, test

def prepare(ds):
    def _map(ex):
        vol = extract_volume(ex["image_name"])
        ex["_valid"] = vol is not None
        ex["_volume"] = vol
        return ex

    ds = ds.map(_map)
    ds = ds.filter(lambda x: x["_valid"])
    return ds
