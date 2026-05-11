import os
from PIL import Image
import pillow_avif

def clean_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except:
                print(f"❌ Removing corrupted file: {file_path}")
                os.remove(file_path)

dataset_path = "datasets/image/test/real"
clean_folder(dataset_path)

print("✅ Dataset cleaned successfully!")