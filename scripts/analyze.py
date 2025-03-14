import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

TACO_DIR = "/Volumes/My Passport/taco"
ANNOTATIONS_FILE = os.path.join(TACO_DIR, "data/annotations.json")
IMAGES_BASE_DIR = os.path.join(TACO_DIR, "data")

with open(ANNOTATIONS_FILE, 'r') as f:
    annotations = json.load(f)

print(f"총 이미지 수: {len(annotations['images'])}")
print(f"총 어노테이션 수: {len(annotations['annotations'])}")
print(f"카테고리 수: {len(annotations['categories'])}")

print("\n카테고리 목록:")
for cat in annotations['categories']:
    print(f"ID: {cat['id']}, 이름: {cat['name']}, 상위 카테고리: {cat.get('supercategory', 'None')}")

img_to_ann = {}
for ann in annotations['annotations']:
    img_id = ann['image_id']
    if img_id not in img_to_ann:
        img_to_ann[img_id] = 0
    img_to_ann[img_id] += 1

category_counts = Counter([ann['category_id'] for ann in annotations['annotations']])
category_names = {cat['id']: cat['name'] for cat in annotations['categories']}

print("\n카테고리별 어노테이션 수:")
for cat_id, count in category_counts.most_common():
    print(f"{category_names[cat_id]}: {count}")

batch_dirs = [d for d in os.listdir(IMAGES_BASE_DIR) if d.startswith('batch_')]
image_count_by_batch = {}

for batch in batch_dirs:
    batch_path = os.path.join(IMAGES_BASE_DIR, batch)
    image_files = [f for f in os.listdir(batch_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_count_by_batch[batch] = len(image_files)

print("\n배치별 이미지 수:")
for batch, count in sorted(image_count_by_batch.items()):
    print(f"{batch}: {count}")

missing_images = 0
for img in annotations['images']:
    filename = img['file_name']
    batch_name = filename.split('/')[0] if '/' in filename else 'batch_1'
    full_path = os.path.join(IMAGES_BASE_DIR, filename)
    
    if not os.path.exists(full_path):
        missing_images += 1
        if missing_images <= 5:  # 처음 5개만
            print(f"누락된 이미지: {full_path}")

print(f"\n누락된 이미지 총 수: {missing_images}")