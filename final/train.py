import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
EPOCHS = 15
TRAIN_IMG_DIR = os.path.join("dataset", "images", "train")
TRAIN_LABEL_DIR = os.path.join("dataset", "labels", "train")

TRASH_CLASSES = [
    "Aluminium foil", "Bottle cap", "Bottle", "Broken glass", "Can", 
    "Carton", "Cigarette", "Cup", "Lid", "Other litter", 
    "Other plastic", "Paper", "Plastic bag - wrapper", "Plastic container", 
    "Pop tab", "Straw", "Styrofoam piece", "Unlabeled litter"
]
NUM_CLASSES = len(TRASH_CLASSES)  # 18개

def load_dataset(img_dir, label_dir):
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    X, y_bbox, y_class = [], [], []
    
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
            y_bbox.append([0, 0, 0, 0])
            y_class.append(0)
            continue
            
        with open(label_files[0], "r") as f:
            line = f.readline().strip()
            parts = line.split()
            if len(parts) < 5:
                y_bbox.append([0, 0, 0, 0])
                y_class.append(0)
            else:
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                y_bbox.append(bbox)
                y_class.append(class_id if class_id < NUM_CLASSES else 0)
    
    X = np.array(X, dtype=np.float32)
    y_bbox = np.array(y_bbox, dtype=np.float32)
    y_class = tf.keras.utils.to_categorical(y_class, num_classes=NUM_CLASSES)
    
    return X, y_bbox, y_class

print("데이터 로드 중.")
X_train, y_train_bbox, y_train_class = load_dataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
print(f"훈련 이미지: {X_train.shape}, 바운딩박스: {y_train_bbox.shape}, 클래스: {y_train_class.shape}")

input_layer = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)

shared_features = Dense(256, activation='relu')(x)
shared_features = Dropout(0.3)(shared_features)

bbox_output = Dense(128, activation='relu')(shared_features)
bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(bbox_output)

class_output = Dense(128, activation='relu')(shared_features)
class_output = Dense(NUM_CLASSES, activation='softmax', name='class_output')(class_output)

model = Model(inputs=input_layer, outputs=[bbox_output, class_output])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss={
        'bbox_output': 'mse',
        'class_output': 'categorical_crossentropy'
    },
    loss_weights={
        'bbox_output': 1.0,
        'class_output': 0.5
    },
    metrics={
        'class_output': 'accuracy'
    }
)

model.summary()

print("모델 학습 시작")
model.fit(
    X_train, 
    {'bbox_output': y_train_bbox, 'class_output': y_train_class},
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1
)

print("모델 저장 중")
model.save("trash_detection_model")

print("ONNX 모델로 변환 중")
import tf2onnx
import onnx

onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, "trash_detection_model.onnx")

print("학습 및 저장 완")