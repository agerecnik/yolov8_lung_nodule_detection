import os
from collections import Counter

def count_images(label_dir):
    counter = Counter()
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            path = os.path.join(label_dir, file)
            if os.path.getsize(path) == 0:
                counter['negatives'] += 1
            else:
                counter['positives'] += 1
    return counter

def print_split_stats(split_name, label_dir):
    counts = count_images(label_dir)
    total = counts['positives'] + counts['negatives']
    print(f"{split_name} set:")
    print(f"  Total images:     {total}")
    print(f"  Positive images:  {counts['positives']}")
    print(f"  Negative images:  {counts['negatives']}")
    print(f"  Positive ratio:   {counts['positives'] / total:.2%}\n")

TRAIN_LABELS = 'yolov8_dataset/labels/train'
VAL_LABELS = 'yolov8_dataset/labels/val'

print_split_stats("Train", TRAIN_LABELS)
print_split_stats("Validation", VAL_LABELS)
