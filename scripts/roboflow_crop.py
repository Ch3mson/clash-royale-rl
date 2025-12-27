import cv2
import os

images_folder = 'dataset/train/images'
labels_folder = 'dataset/train/labels'
os.makedirs('crops', exist_ok=True)

count = 0
for label_file in os.listdir(labels_folder):
    img_name = label_file.replace('.txt', '.jpg')
    img = cv2.imread(f'{images_folder}/{img_name}')
    h, w = img.shape[:2]
    
    with open(f'{labels_folder}/{label_file}') as f:
        for line in f:
            parts = line.strip().split()
            # YOLO format: class x_center y_center width height (normalized)
            cx, cy, bw, bh = map(float, parts[1:5])
            
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            crop = img[y1:y2, x1:x2]
            cv2.imwrite(f'crops/crop_{count:04d}.png', crop)
            count += 1

print(f'Saved {count} crops')