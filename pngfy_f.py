import shutil
import cv2
import os

def crop_and_save_signature(input_path, output_path):
    size = 64
    target_size=56
    img = Image.open(input_path).convert('L')  # grayscale
    arr = np.array(img)
    _, binary_arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find non-white pixels (threshold 240 to handle noise)
    coords = np.where(binary_arr < 240)
    if len(coords[0]) == 0:
        img.resize((size, size), Image.Resampling.LANCZOS).save(output_path)
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
    if input_path !== ""
        augment = T.Compose([
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomRotation(degrees=15),
            T.RandomPerspective(distortion_scale=0.4, p=0.4),
            # RandomErasing not useful here (it's tensor-only)
        ])
        
        cropped = augment(cropped)
    # Resize cropped signature to exactly target_size × target_size (preserving aspect ratio)
    cropped = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
        # Resize to 64x64 (keeps aspect ratio with white padding if needed)
    
    bg = Image.new('L', (size, size), 255)
    
    # Center it
    x = (size - target_size) // 2
    y = (size - target_size) // 2
    bg.paste(cropped, (x, y))

    bg.save(output_path, "PNG", optimize=True)

# Run on your dataset
def pngfy(directories):
    print(directories)
    for input_dir, output_dir in directories:
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)   # ← nukes everything
            print(f"Deleted old folder: {output_dir}" )
        os.makedirs(output_dir, exist_ok=True)
        
        for file in os.listdir(input_dir):
                if file.endswith(('.PNG')):
                    crop_and_save_signature(os.path.join(input_dir, file),
                                        os.path.join(output_dir, file.split('.')[0] + ".PNG"))
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    crop_and_save_signature(os.path.join(input_dir, file),
                                        os.path.join(output_dir, file.split('.')[0] + ".png"))
    
    print("All datasets processed perfectly!")

print("All the datasets are processed perfectly!")