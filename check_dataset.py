
import os

DATASET_PATH = "dataset/PlantVillage"

print("Checking PlantVillage dataset...")

if not os.path.exists(DATASET_PATH):
    print("Dataset folder not found!")
    exit()

classes = [
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
]

total_images = 0

for class_name in sorted(classes):
    folder_path = os.path.join(DATASET_PATH, class_name)

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(class_name, ":", len(images), "images")
    total_images += len(images)

print("\nTotal classes:", len(classes))
print("Total images:", total_images)
print("Dataset verification completed!")
