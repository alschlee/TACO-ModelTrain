import json
import os
import shutil
import numpy as np
from PIL import Image
import random

TACO_DIR = "/Volumes/My Passport/taco"
ANNOTATIONS_FILE = os.path.join(TACO_DIR, "data/annotations.json")
IMAGES_BASE_DIR = os.path.join(TACO_DIR, "data")
OUTPUT_DIR = os.path.join(TACO_DIR, "processed_data")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("어노테이션 파일 로드 중")
with open(ANNOTATIONS_FILE, 'r') as f:
    annotations = json.load(f)

categories = {cat['id']: cat['name'] for cat in annotations['categories']}
category_id_to_idx = {cat['id']: i for i, cat in enumerate(annotations['categories'])}

with open(os.path.join(OUTPUT_DIR, 'classes.txt'), 'w') as f:
    for i, cat in enumerate(sorted(annotations['categories'], key=lambda x: category_id_to_idx[x['id']])):
        f.write(f"{cat['name']}\n")

print("이미지와 어노테이션 매핑 중")
image_to_annotations = {}
for ann in annotations['annotations']:
    img_id = ann['image_id']
    if img_id not in image_to_annotations:
        image_to_annotations[img_id] = []
    image_to_annotations[img_id].append(ann)

image_id_to_info = {img['id']: img for img in annotations['images']}

print("데이터셋 분할 중")
image_ids = list(set(img_id for img_id in image_to_annotations.keys()))
random.shuffle(image_ids)

n_total = len(image_ids)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_image_ids = image_ids[:n_train]
val_image_ids = image_ids[n_train:n_train+n_val]
test_image_ids = image_ids[n_train+n_val:]

print(f"학습 이미지 수: {len(train_image_ids)}")
print(f"검증 이미지 수: {len(val_image_ids)}")
print(f"테스트 이미지 수: {len(test_image_ids)}")

def process_dataset(image_ids, output_subdir, format_type="yolo"):

    processed_count = 0
    
    for img_id in image_ids:
        if img_id not in image_id_to_info:
            continue
            
        img_info = image_id_to_info[img_id]
        filename = img_info['file_name']
        
        if '/' in filename:
            img_path = os.path.join(IMAGES_BASE_DIR, filename)
        else:
            batch_dirs = [d for d in os.listdir(IMAGES_BASE_DIR) if d.startswith('batch_')]
            found = False
            for batch in batch_dirs:
                test_path = os.path.join(IMAGES_BASE_DIR, batch, filename)
                if os.path.exists(test_path):
                    img_path = test_path
                    found = True
                    break
            
            if not found:
                print(f"Warning: {filename} not found, skipping.")
                continue
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue
        
        output_img_path = os.path.join(OUTPUT_DIR, output_subdir, os.path.basename(filename))
        shutil.copy(img_path, output_img_path)
        
        with Image.open(img_path) as img:
            width, height = img.size
        
        if img_id in image_to_annotations:
            if format_type == "yolo":
                boxes = []
                for ann in image_to_annotations[img_id]:
                    category_idx = category_id_to_idx[ann['category_id']]
                    x, y, w, h = ann['bbox']
                    
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    norm_width = w / width
                    norm_height = h / height
                    
                    boxes.append(f"{category_idx} {x_center} {y_center} {norm_width} {norm_height}")
                
                basename = os.path.splitext(os.path.basename(filename))[0]
                with open(os.path.join(OUTPUT_DIR, output_subdir, f"{basename}.txt"), 'w') as f:
                    f.write("\n".join(boxes))
            
            elif format_type == "voc":
                pass
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"{output_subdir}: {processed_count}/{len(image_ids)} 처리 완료")

print("학습 데이터 처리 중")
process_dataset(train_image_ids, "train", "yolo")
print("검증 데이터 처리 중")
process_dataset(val_image_ids, "val", "yolo")
print("테스트 데이터 처리 중")
process_dataset(test_image_ids, "test", "yolo")

print(f"데이터 전처리 완료~~~ 저장: {OUTPUT_DIR}")

with open(os.path.join(OUTPUT_DIR, 'dataset_stats.txt'), 'w') as f:
    f.write(f"총 이미지 수: {n_total}\n")
    f.write(f"학습 이미지 수: {len(train_image_ids)}\n")
    f.write(f"검증 이미지 수: {len(val_image_ids)}\n")
    f.write(f"테스트 이미지 수: {len(test_image_ids)}\n")
    f.write(f"카테고리 수: {len(annotations['categories'])}\n")