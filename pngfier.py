import os
from PIL import Image
import numpy as np
import shutil

def crop_and_save_signature(input_path, output_path):
    img = Image.open(input_path).convert('L')  # grayscale
    arr = np.array(img)

    # Find non-white pixels (threshold 240 to handle noise)
    coords = np.where(arr < 240)
    if len(coords[0]) == 0:
        img.save(output_path)
        return

    y0, y1 = coords[0].min(), coords[0].max() + 1
    x0, x1 = coords[1].min(), coords[1].max() + 1

    # Add 10% padding (prevents cutting edges)
    h, w = y1 - y0, x1 - x0
    pad_y = int(h * 0.1)
    pad_x = int(w * 0.1)

    y0 = max(0, y0 - pad_y)
    y1 = min(arr.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(arr.shape[1], x1 + pad_x)

    cropped = img.crop((x0, y0, x1, y1))
    size = 64
    # Resize to 64x64 (keeps aspect ratio with white padding if needed)
    background = Image.new('L', (size, size), 255)  # white background
    cropped.thumbnail((size-4 , size-4), Image.Resampling.LANCZOS)  # slight shrink to fit
    background.paste(cropped, ((size - cropped.width) // 2, (size - cropped.height) // 2))

    background.save(output_path, "PNG", optimize=True)

# Run on your dataset
def pngfy(directories):
    
    for input_dir, output_dir in directories:
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)   # ← nukes everything
            print(f"Deleted old folder: {output_dir}" )
        os.makedirs(output_dir, exist_ok=True)
        
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG')):
                crop_and_save_signature(os.path.join(input_dir, file),
                                        os.path.join(output_dir, file.split('.')[0] + ".png"))
    
    print("All datasets processed perfectly!")



def crop_and_save_signature(input_path, output_path, size=64, target_size=56):
    img = Image.open(input_path).convert('L')
    arr = np.array(img)

    coords = np.where(arr < 240)
    if len(coords[0]) == 0:
        img.resize((size, size), Image.Resampling.LANCZOS).save(output_path)
        return

    y0, y1 = coords[0].min(), coords[0].max() + 1
    x0, x1 = coords[1].min(), coords[1].max() + 1

    h, w = y1 - y0, x1 - x0
    pad_y = max(int(h * 0.1), 10)
    pad_x = max(int(w * 0.1), 10)

    y0 = max(0, y0 - pad_y)
    y1 = min(arr.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(arr.shape[1], x1 + pad_x)

    cropped = img.crop((x0, y0, x1, y1))

    # CRITICAL FIX: Resize to EXACT target size (keeps aspect ratio via padding)
    bg = Image.new('L', (size, size), 255)
    
    # Resize cropped signature to exactly target_size × target_size (preserving aspect ratio)
    cropped = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Center it
    x = (size - target_size) // 2
    y = (size - target_size) // 2
    bg.paste(cropped, (x, y))

    bg.save(output_path, "PNG", optimize=True)