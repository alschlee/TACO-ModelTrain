import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import Adam

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
EPOCHS = 10
TRAIN_IMG_DIR = os.path.join("dataset", "images", "train")
TRAIN_LABEL_DIR = os.path.join("dataset", "labels", "train")

def load_dataset(img_dir, label_dir):
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    X, y = [], []
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X.append(img / 255.0)
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_files = glob.glob(os.path.join(label_dir, f"{base_name}*.txt"))
        if not label_files:
            y.append([0, 0, 0, 0])
            continue

        with open(label_files[0], "r") as f:
            line = f.readline().strip()
            parts = line.split()
            if len(parts) < 5:
                y.append([0, 0, 0, 0])
            else:
                bbox = list(map(float, parts[1:5]))
                y.append(bbox)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

print("데이터 로드 중")
X_train, y_train = load_dataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
print(f"훈련 이미지: {X_train.shape}, 라벨: {y_train.shape}")

model = Sequential([
    InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
model.summary()

print("모델 학습 시작")
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

print("모델 저장 중")
model.export("saved_model")
print("학습 및 저장 완")
