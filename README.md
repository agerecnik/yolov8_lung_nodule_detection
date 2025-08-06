# YOLOv8-Based Lung Nodule Detection on LUNA16

This project demonstrates how to train [YOLOv8](https://github.com/ultralytics/ultralytics) to detect lung nodules in CT scans from the [LUNA16 dataset](https://luna16.grand-challenge.org/). It includes scripts for converting 3D volumetric CT data to 2D image slices, generating YOLO annotations, dataset splitting, visual validation, and training.

---

## Repository Structure

```
.
├── input_data/                     # Folder for downloaded LUNA16 .mhd files
├── yolov8_dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── preprocess.py                  # Script to convert 3D scans to YOLO-compatible 2D images + labels
├── split_datasets.py             # Script to split dataset into train/val with balanced positive/negative slices
├── check_split_distribution.py   # Utility script to verify class distribution in the split
├── train_yolov8.py               # YOLOv8 training script
├── yolov8.yaml                   # Dataset configuration file for YOLOv8
├── visualize_annotations_and_bounding_boxes.ipynb  # Jupyter notebook to verify bounding boxes
├── requirements.txt
└── README.md
```

---

## Quick Start: Using Preprocessed Data

If you'd like to skip data preparation, the folder `yolov8_dataset/` already contains:
- Preprocessed 2D CT slices in `images/train/` and `images/val/`
- Corresponding YOLO format labels in `labels/train/` and `labels/val/`

You can jump directly to training (see below).

---

## Full Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/luna16-yolov8-nodule-detection.git
cd luna16-yolov8-nodule-detection
```

### 2. Download LUNA16 3D CT scans

Download all `.mhd` files from the official LUNA16 website and place them in a folder called `input_data/`.

### 3. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Preprocess the data

Convert 3D CT volumes into 2D axial slices with YOLOv8-compatible bounding box annotations:

```bash
python preprocess.py
```

- Positive slices include bounding boxes converted from nodule center coordinates and diameters.
- Negative slices are selected randomly from slices without nodules (with a margin buffer).
- You can modify logic inside the script to select more/less slices per nodule.

### 5. Visual verification (optional)

Launch Jupyter Notebook to visually inspect annotations and YOLO bounding boxes:

```bash
jupyter notebook
```

Open `visualize_annotations_and_bounding_boxes.ipynb` and scroll through scans with bounding boxes and annotations overlaid.

<img width="371" height="355" alt="Visual verification" src="https://github.com/user-attachments/assets/e93993ab-23eb-4093-be50-6c7fa531281a" />


### 6. Split dataset

```bash
python split_datasets.py
```

This creates training and validation sets with a balanced number of positive and negative examples. All positive and negative slices taken from the same CT series are put in either train or val dataset to prevent data leakage.

To check class balance:

```bash
python check_split_distribution.py
```

---

## Model Training

Edit `train_yolov8.py` to select the desired model size (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`):

```python
model = YOLO("yolov8m.pt")
```

Then start training:

```bash
python train_yolov8.py
```

Training settings:
- Input size: 512×512
- Augmentations are adapted for CT (no color transforms, no mosaic/mixup)
- Batch size: 32
- Epochs: 300

---

## Sample Results
The following results were achieved using the included preprocessed images and the following hardware:
-   **GPU**: NVIDIA RTX 2080 Ti, 11 GB GDDR6,
-   **CPU**: Intel Xeon W-2155, 10 core / 20 threads, up to 4.5 GHz,
-   **RAM**: 64 GB DDR4,

| Model     | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Training Time |
|-----------|-----------|--------|---------|--------------|----------------|
| YOLOv8n   | 0.825     | 0.679  | 0.774   | 0.443        | 1.7 h          |
| YOLOv8s   | 0.837     | 0.696  | 0.791   | 0.453        | 2.9 h          |
| YOLOv8m   | 0.851     | 0.705  | 0.821   | 0.475        | 6.2 h          |
| YOLOv8l   | 0.815     | 0.763  | 0.819   | 0.482        | 9.1 h          |

---

## Future Improvements

### 1. **Manual Annotation Per Slice**

Currently, bounding boxes in neighboring slices are simply duplicated from the central slice where the nodule is centered. While this provides approximate spatial context, it is not precise, especially for small nodules that may disappear entirely in adjacent slices. A more accurate method would involve:

- Manually inspecting and labeling each individual slice that contains a visible part of the nodule.

- Adjusting the bounding box size and position accordingly per slice.  

This would eliminate false positives in slices where the nodule is no longer visible and improve label quality for training.
    

### 2. **Use of Full LIDC-IDRI Dataset**

The LUNA16 challenge dataset is a subset of the larger LIDC-IDRI dataset. By including additional scans from the remaining patients:

- The training set could be expanded significantly, especially with more diverse and rare nodule types.
    
- It would reduce overfitting and improve generalization, particularly for larger models like YOLOv8m and YOLOv8l.
    

### 3. **Improved Train and Validation Dataset Splitting**

In the current implementation, the dataset is split to balance positive and negative slices. All positive and negative slices taken from the same CT series are put in either train or val dataset to prevent data leakage, but it does not account for:

- Nodule size (e.g., small vs. large)
    
- Scan origin (some patients may have multiple series)
    
A more robust stratified sampling strategy, possibly even patient-wise, could lead to more reliable validation results and better model generalization.

### 4. **Lung Region Segmentation (Preprocessing)**

Many slices in CT scans contain irrelevant anatomy (e.g., outside the lungs), leading to noisy backgrounds and false positives. An effective solution would be to apply a lung segmentation model to each slice (or 3D volume) and mask out areas outside the lung fields before training or prediction. This preprocessing step would allow the model to focus on relevant anatomy and reduce background noise during learning.
