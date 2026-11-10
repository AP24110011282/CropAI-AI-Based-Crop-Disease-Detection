
import tensorflow as tf
import os
import json

DATASET_PATH = "dataset/PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("Starting image preprocessing...", flush=True)

# Load training dataset (70%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Load remaining dataset (30%)
val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Divide remaining data into validation and testing
val_batches = tf.data.experimental.cardinality(val_test_ds)
test_ds = val_test_ds.take(val_batches // 2)
val_ds = val_test_ds.skip(val_batches // 2)

# Save class names
class_names = train_ds.class_names

os.makedirs("models", exist_ok=True)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("\nPreprocessing completed!", flush=True)
print("Number of classes:", len(class_names))
print("Class names:", class_names)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("\nTraining, validation and testing datasets are ready.")
