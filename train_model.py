
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = "dataset/PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Starting Crop Disease Detection Training...")

# ==========================================
# 2. COLLECT IMAGE PATHS AND LABELS
# ==========================================

dataset_dir = Path(DATASET_PATH)

class_names = sorted([
    folder.name for folder in dataset_dir.iterdir()
    if folder.is_dir()
])

if len(class_names) == 0:
    raise ValueError("No class folders found in dataset/PlantVillage")

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

image_extensions = {".jpg", ".jpeg", ".png"}

image_paths = []
labels = []

for class_name in class_names:
    class_folder = dataset_dir / class_name

    for image_path in class_folder.rglob("*"):
        if image_path.suffix.lower() in image_extensions:
            image_paths.append(str(image_path))
            labels.append(class_to_index[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels)

num_classes = len(class_names)

print("Total images:", len(image_paths))
print("Total classes:", num_classes)

# ==========================================
# 3. STRATIFIED DATASET SPLITTING
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("\nDataset split:")
print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))

# Save class names
with open("models/class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

# ==========================================
# 4. CREATE TENSORFLOW DATASETS
# ==========================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)

    image = preprocess_input(image)

    label = tf.one_hot(label, num_classes)

    return image, label


AUTOTUNE = tf.data.AUTOTUNE


def create_dataset(paths, dataset_labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, dataset_labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = create_dataset(X_train, y_train, training=True)
val_ds = create_dataset(X_val, y_val)
test_ds = create_dataset(X_test, y_test)

# ==========================================
# 5. CLASS WEIGHTS
# ==========================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(weights)
}

# ==========================================
# 6. LOAD MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ==========================================
# 7. BUILD MODEL
# ==========================================

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        num_classes,
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 8. CALLBACKS
# ==========================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_crop_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nStarting model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

# ==========================================
# 10. SAVE MODEL
# ==========================================

model.save("models/crop_disease_model.keras")

print("\nModel saved successfully!")

# ==========================================
# 11. TEST MODEL
# ==========================================

print("\nEvaluating test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

# ==========================================
# 12. SAVE TRAINING HISTORY
# ==========================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open("reports/training_history.json", "w") as file:
    json.dump(history_data, file, indent=4)

# ==========================================
# 13. ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Crop Disease Detection Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("reports/accuracy.png")
plt.close()

# ==========================================
# 14. LOSS GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Crop Disease Detection Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("reports/loss.png")
plt.close()

print("\n================================")
print("MODEL TRAINING COMPLETED!")
print("================================")
print("Classes:", num_classes)
print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))
print("Test accuracy:", round(test_accuracy * 100, 2), "%")
print("Reports saved in reports folder.")
