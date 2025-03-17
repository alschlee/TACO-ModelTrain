import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# --- 설정 ---
MODEL_PATH = "/Volumes/My Passport/TrashDetect/saved_model.keras"
IMG_HEIGHT, IMG_WIDTH = 224, 224
TEST_IMG_DIR = os.path.join("dataset", "images", "test")
TEST_LABEL_DIR = os.path.join("dataset", "labels", "test")

# 실제 데이터셋의 클래스 정의
TRASH_CLASSES = [
    "Aluminium foil", "Bottle cap", "Bottle", "Broken glass", "Can", 
    "Carton", "Cigarette", "Cup", "Lid", "Other litter", 
    "Other plastic", "Paper", "Plastic bag - wrapper", "Plastic container", 
    "Pop tab", "Straw", "Styrofoam piece", "Unlabeled litter"
]
NUM_CLASSES = len(TRASH_CLASSES)

# --- 데이터 로드 함수 ---
def load_dataset(img_dir, label_dir):
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    X, y_class = [], []
    
    for img_path in img_paths:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        X.append(img)
        
        # 라벨 파일 찾기
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_files = glob.glob(os.path.join(label_dir, f"{base_name}*.txt"))
        
        if not label_files:
            y_class.append(0)
            continue
        
        with open(label_files[0], "r") as f:
            line = f.readline().strip()
            parts = line.split()
            class_id = int(parts[0]) if len(parts) >= 1 else 0
            y_class.append(class_id if class_id < NUM_CLASSES else 0)
    
    X = np.array(X, dtype=np.float32)
    y_class = np.array(y_class)
    y_class_one_hot = tf.keras.utils.to_categorical(y_class, num_classes=NUM_CLASSES)
    
    return X, y_class, y_class_one_hot

# --- 모델 평가 ---
print("테스트 데이터 로드 중")
X_test, y_test, y_test_one_hot = load_dataset(TEST_IMG_DIR, TEST_LABEL_DIR)
print(f"테스트 데이터: {X_test.shape}, 클래스 라벨: {y_test.shape}")

# 모델 로드
print("모델 로드 중")
model = load_model(MODEL_PATH)

# 예측 수행
print("모델 평가 중")
class_predictions = model.predict(X_test)

# 원-핫에서 클래스 ID로 변환
y_pred = np.argmax(class_predictions, axis=1)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"모델 정확도: {accuracy:.2f}%")