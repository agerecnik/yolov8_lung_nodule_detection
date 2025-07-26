import os
import pandas as pd
from glob import glob
import numpy as np
from shutil import copy2
from tqdm import tqdm

# --- CONFIG ---
ANNOTATION_FILE = "annotations.csv"
IMAGE_POS_DIR = "yolov8_dataset/images/positives"
IMAGE_NEG_DIR = "yolov8_dataset/images/negatives"
LABEL_POS_DIR = "yolov8_dataset/labels/positives"
LABEL_NEG_DIR = "yolov8_dataset/labels/negatives"

IMAGE_TRAIN_DIR = "yolov8_dataset/images/train"
IMAGE_VAL_DIR = "yolov8_dataset/images/val"
LABEL_TRAIN_DIR = "yolov8_dataset/labels/train"
LABEL_VAL_DIR = "yolov8_dataset/labels/val"

for d in [IMAGE_TRAIN_DIR, IMAGE_VAL_DIR, LABEL_TRAIN_DIR, LABEL_VAL_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Load annotated series ---
annotated_series = set(pd.read_csv(ANNOTATION_FILE)['seriesuid'].unique())

# --- Get UIDs from positive and negative images ---
def get_series_uids(image_dir):
    return set(os.path.basename(f).split('_')[0] for f in glob(os.path.join(image_dir, '*.png')))

all_pos_series = get_series_uids(IMAGE_POS_DIR)
all_neg_series = get_series_uids(IMAGE_NEG_DIR)

# --- Separate annotated vs non-annotated (based on .csv) ---
annotated_neg_series = all_neg_series & annotated_series
nonannotated_neg_series = all_neg_series - annotated_series

# --- Split series ---
def split_series(series_set, train_ratio=0.8):
    series_list = sorted(series_set)
    np.random.shuffle(series_list)
    split = int(len(series_list) * train_ratio)
    return set(series_list[:split]), set(series_list[split:])

train_annotated, val_annotated = split_series(all_pos_series)
train_nonannotated, val_nonannotated = split_series(nonannotated_neg_series)

# --- Combine to final splits ---
train_uids = train_annotated | train_nonannotated
val_uids = val_annotated | val_nonannotated

# --- Copy function ---
def copy_split_files(series_uids, img_dirs, lbl_dirs, img_out_dir, lbl_out_dir):
    for uid in tqdm(series_uids, desc=f"Copying {img_out_dir}"):
        for img_dir, lbl_dir in zip(img_dirs, lbl_dirs):
            for img_path in glob(os.path.join(img_dir, f"{uid}_*.png")):
                base = os.path.basename(img_path)
                label_path = os.path.join(lbl_dir, base.replace(".png", ".txt"))

                copy2(img_path, os.path.join(img_out_dir, base))
                if os.path.exists(label_path):
                    copy2(label_path, os.path.join(lbl_out_dir, base.replace(".png", ".txt")))
                else:
                    open(os.path.join(lbl_out_dir, base.replace(".png", ".txt")), 'w').close()

# --- Copy all matching images and labels ---
copy_split_files(train_uids, [IMAGE_POS_DIR, IMAGE_NEG_DIR], [LABEL_POS_DIR, LABEL_NEG_DIR], IMAGE_TRAIN_DIR, LABEL_TRAIN_DIR)
copy_split_files(val_uids, [IMAGE_POS_DIR, IMAGE_NEG_DIR], [LABEL_POS_DIR, LABEL_NEG_DIR], IMAGE_VAL_DIR, LABEL_VAL_DIR)

# --- Summary ---
print(f"\nTrain series: {len(train_uids)} (Annotated: {len(train_annotated)}, Non-Annotated: {len(train_nonannotated)})")
print(f"Val series:   {len(val_uids)} (Annotated: {len(val_annotated)}, Non-Annotated: {len(val_nonannotated)})")
