import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from glob import glob
from tqdm import tqdm

# --- CONFIG ---
RESOURCES_PATH = os.path.abspath('input_data')
ANNOTATION_FILE = os.path.abspath('annotations.csv')
OUTPUT_IMAGES_POS = os.path.abspath('yolov8_dataset/images/positives')
OUTPUT_LABELS_POS = os.path.abspath('yolov8_dataset/labels/positives')
OUTPUT_IMAGES_NEG = os.path.abspath('yolov8_dataset/images/negatives')
OUTPUT_LABELS_NEG = os.path.abspath('yolov8_dataset/labels/negatives')

os.makedirs(OUTPUT_IMAGES_POS, exist_ok=True)
os.makedirs(OUTPUT_LABELS_POS, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_NEG, exist_ok=True)
os.makedirs(OUTPUT_LABELS_NEG, exist_ok=True)

# --- Load annotations ---
df = pd.read_csv(ANNOTATION_FILE)
all_uids_with_nodules = set(df['seriesuid'].unique())

def world_to_voxel(world, origin, spacing):
    stretched = np.abs(np.array(world) - np.array(origin))
    voxel = stretched / spacing
    return voxel.astype(int)

def load_ct(seriesuid):
    path = glob(os.path.join(RESOURCES_PATH, '*', f'{seriesuid}.mhd'))[0]
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)  # shape: [z, y, x]
    spacing = np.array(image.GetSpacing())[::-1]  # [z, y, x]
    origin = np.array(image.GetOrigin())[::-1]
    return array, spacing, origin

def save_slice_and_labels(seriesuid, z, slice_img, label_lines, image_dir, label_dir):
    img_filename = f'{seriesuid}_{z}.png'
    label_filename = f'{seriesuid}_{z}.txt'

    cv2.imwrite(os.path.join(image_dir, img_filename), slice_img)
    with open(os.path.join(label_dir, label_filename), 'w') as f:
        for line in label_lines:
            f.write(line + '\n')

# --- Process annotated series ---
print("Processing annotated series...")
grouped = df.groupby("seriesuid")

for seriesuid, group in tqdm(grouped, total=len(grouped)):
    try:
        image, spacing, origin = load_ct(seriesuid)
        forbidden_slices = set()
        labels_by_slice = {}

        for _, row in group.iterrows():
            world_coord = [row['coordZ'], row['coordY'], row['coordX']]
            diameter = row['diameter_mm']
            voxel = world_to_voxel(world_coord, origin, spacing)
            z, y, x = voxel

            if z < 0 or z >= image.shape[0]:
                continue

            radius_px = diameter / spacing[1] / 2
            h, w = image.shape[1:]
            x_center = x / w
            y_center = y / h
            box_margin = 1.35
            box_w = box_h = (radius_px * 2 * box_margin) / w

            line = f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            # Add center slice
            labels_by_slice.setdefault(z, []).append(line)

            # Add the first slice below (z-1)
            if diameter >= 8 and z - 1 >= 0:
                labels_by_slice.setdefault(z - 1, []).append(line)

            # Add the first slice above (z+1)
            if z + 1 < image.shape[0]:
                    labels_by_slice.setdefault(z + 1, []).append(line)
            
            # Add the second slice above (z+2)            
            if diameter >= 8 and z + 2 < image.shape[0]:
                labels_by_slice.setdefault(z + 2, []).append(line)

            # Compute forbidden slice range based on z-spacing + extra buffer
            z_radius = diameter / (2 * spacing[0])
            margin = int(np.ceil(z_radius + 5))  # nodule radius + 5-slice safety buffer
            forbidden_range = range(max(0, z - margin), min(image.shape[0], z + margin + 1))
            forbidden_slices.update(forbidden_range)

        for z, lines in labels_by_slice.items():
            slice_img = image[z]
            slice_img = np.clip(slice_img, -1000, 400)
            slice_img = ((slice_img + 1000) / 1400 * 255).astype(np.uint8)
            save_slice_and_labels(seriesuid, z, slice_img, lines, OUTPUT_IMAGES_POS, OUTPUT_LABELS_POS)

        # Safe negative sampling: avoid forbidden slices
        all_slices = set(range(image.shape[0]))
        negative_candidates = list(all_slices - forbidden_slices)
        np.random.shuffle(negative_candidates)

        for z in negative_candidates[:3]:
            slice_img = image[z]
            slice_img = np.clip(slice_img, -1000, 400)
            slice_img = ((slice_img + 1000) / 1400 * 255).astype(np.uint8)
            save_slice_and_labels(seriesuid, z, slice_img, [], OUTPUT_IMAGES_NEG, OUTPUT_LABELS_NEG)

    except Exception as e:
        print(f"Error processing {seriesuid}: {e}")
        
# --- Process non-annotated series ---
print("Processing non-annotated series...")
all_mhd_files = glob(os.path.join(RESOURCES_PATH, '*', '*.mhd'))
all_uids = [os.path.splitext(os.path.basename(p))[0] for p in all_mhd_files]
non_annotated_uids = [uid for uid in all_uids if uid not in all_uids_with_nodules]

for seriesuid in tqdm(non_annotated_uids):
    try:
        image, spacing, origin = load_ct(seriesuid)
        total_slices = image.shape[0]
        candidates = list(range(total_slices))
        np.random.shuffle(candidates)

        for z in candidates[:3]:
            slice_img = image[z]
            slice_img = np.clip(slice_img, -1000, 400)
            slice_img = ((slice_img + 1000) / 1400 * 255).astype(np.uint8)
            save_slice_and_labels(seriesuid, z, slice_img, [], OUTPUT_IMAGES_NEG, OUTPUT_LABELS_NEG)

    except Exception as e:
        print(f"Error processing {seriesuid}: {e}")
