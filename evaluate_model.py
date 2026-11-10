
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = "dataset/PlantVillage"
MODEL_PATH = "models/crop_disease_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

os.makedirs("reports", exist_ok=True)

print("Loading trained model...")

# ==========================================
# 2. LOAD MODEL AND CLASS NAMES
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

with open("models/class_names.json", "r") as file:
    class_names = json.load(file)

num_classes = len(class_names)

print("Number of classes:", num_classes)

# ==========================================
# 3. COLLECT DATASET
# ==========================================

dataset_dir = Path(DATASET_PATH)

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

# ==========================================
# 4. RECREATE EXACT SAME TEST SPLIT
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

print("\nTest images:", len(X_test))

# ==========================================
# 5. CREATE TEST DATASET
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


test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ==========================================
# 6. GENERATE PREDICTIONS
# ==========================================

y_true = []
y_pred = []

print("\nGenerating predictions...")

for images, labels_batch in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        np.argmax(labels_batch.numpy(), axis=1)
    )

    y_pred.extend(
        np.argmax(predictions, axis=1)
    )

# ==========================================
# 7. ACCURACY
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0,
    output_dict=True
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
)

with open(
    "reports/classification_report.json",
    "w"
) as file:

    json.dump(report, file, indent=4)

# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

plt.figure(figsize=(17, 14))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Crop Disease Detection - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================================
# 10. SAVE EVALUATION SUMMARY
# ==========================================

summary = {
    "test_accuracy": float(accuracy),
    "test_accuracy_percentage": round(float(accuracy) * 100, 2),
    "total_test_images": len(y_true),
    "number_of_classes": num_classes,
    "class_names": class_names
}

with open(
    "reports/evaluation_summary.json",
    "w"
) as file:

    json.dump(summary, file, indent=4)

# ==========================================
# 11. VERIFY CLASS SUPPORT
# ==========================================

print("\nTest images per class:")

for index, class_name in enumerate(class_names):

    count = int(np.sum(np.array(y_true) == index))

    print(class_name, ":", count)

    if count == 0:
        print("WARNING: No test images for this class!")

# ==========================================
# 12. FINAL RESULT
# ==========================================

print("\n================================")
print("MODEL EVALUATION COMPLETED!")
print("================================")

print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("Total Test Images:", len(y_true))
print("Total Classes:", num_classes)

print("\nReports saved in reports folder.")
