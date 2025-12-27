# Training YOLOv8 Card Hand Classifier

This guide explains how to train a YOLOv8 **image classification** model to identify cards in your hand, replacing the current template matching approach.

---

## Why YOLOv8 Classification vs Template Matching?

**Current Approach (Template Matching)**:
- ✅ Fast (~5ms per card)
- ✅ No training required
- ❌ Requires manual template collection for each card variant
- ❌ Struggles with similar-looking cards
- ❌ 86.7% detection rate (52/60 cards)

**YOLOv8 Classification**:
- ✅ Much more accurate (>95% typical)
- ✅ Handles variations automatically (different levels, skins, lighting)
- ✅ Better with similar-looking cards
- ❌ Requires training data collection
- ❌ Slightly slower (~15ms per card, still very fast)

---

## Overview: What You're Training

**Model Type**: YOLOv8 Image Classification (not object detection!)

**Input**: Cropped card image from hand slot (136x185 pixels)

**Output**: Card name with confidence score (e.g., "wizard" 0.98)

**Training Time**:
- Google Colab (free T4 GPU): ~10-15 minutes
- Local CPU: ~2-3 hours (not recommended)
- Local GPU (if available): ~20-30 minutes

---

## Step 1: Collect Training Data

You need images of each card in your hand. There are 3 methods:

### Method 1: Use Existing Template Images (Quick Start)

You already have templates in `detection/card_templates/`. Convert them to a dataset:

```bash
python3 scripts/convert_templates_to_dataset.py
```

This will create a dataset in `datasets/card_hand/` using your existing templates.

**Pros**: Quick to get started
**Cons**: Limited data (~5 images per card), may not generalize well

### Method 2: Auto-Collect During Gameplay (Recommended)

Run the bot with screenshot collection enabled:

```bash
python3 main.py --screenshots --games 10
```

This saves card crops to `training_data/session_<timestamp>/card_crops/`

After collecting, organize them:

```bash
python3 scripts/organize_card_data.py training_data/session_*/card_crops/
```

**Pros**: Real gameplay data, good variety
**Cons**: Requires manual labeling of unknown cards

### Method 3: Manual Screenshot Collection

1. Play the game normally
2. Take screenshots when you have the card you want
3. Use the crop tool to extract cards:

```python
from detection.card_hand_detector import CardHandDetector

detector = CardHandDetector()
screenshot = cv2.imread('your_screenshot.png')
detector.save_card_crops(screenshot, output_dir='manual_crops')
```

---

## Step 2: Organize Dataset

Your dataset should follow this structure:

```
datasets/card_hand_classifier/
├── train/
│   ├── archers/
│   │   ├── img001.png
│   │   ├── img002.png
│   │   └── ...
│   ├── wizard/
│   │   ├── img001.png
│   │   └── ...
│   ├── giant/
│   └── ... (one folder per card)
├── val/
│   ├── archers/
│   ├── wizard/
│   └── ...
└── test/
    ├── archers/
    ├── wizard/
    └── ...
```

**Split Ratios**:
- Train: 70% of images
- Validation: 20% of images
- Test: 10% of images

**Minimum Images per Card**:
- Ideal: 50+ images per card
- Acceptable: 20+ images per card
- Minimum: 10+ images per card

Use the provided script to auto-split:

```bash
python3 scripts/split_dataset.py datasets/raw_card_images/ datasets/card_hand_classifier/
```

---

## Step 3: Train on Google Colab (Recommended)

### 3.1: Upload Dataset to Google Drive

1. Zip your dataset:
```bash
cd datasets
zip -r card_hand_classifier.zip card_hand_classifier/
```

2. Upload `card_hand_classifier.zip` to Google Drive

### 3.2: Open Google Colab

Open this notebook in Colab: [YOLOv8 Classification Training](https://colab.research.google.com/)

### 3.3: Run Training Code

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install Ultralytics
!pip install ultralytics

# Unzip dataset
!unzip /content/drive/MyDrive/card_hand_classifier.zip -d /content/

# Import YOLO
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n-cls.pt')  # nano (fastest)
# model = YOLO('yolov8s-cls.pt')  # small (more accurate)
# model = YOLO('yolov8m-cls.pt')  # medium (best balance)

# Train
results = model.train(
    data='/content/card_hand_classifier',
    epochs=50,              # Number of training iterations
    imgsz=224,              # Image size (224 is standard for classification)
    batch=32,               # Batch size (reduce if GPU memory issues)
    patience=10,            # Early stopping if no improvement
    save=True,              # Save best model
    project='card_classifier',
    name='run1',
    verbose=True
)

# Validate
results = model.val()

# Export best model
model.export(format='torchscript')  # or 'onnx' for faster inference
```

### 3.4: Download Trained Model

After training completes, download:
- `card_classifier/run1/weights/best.pt` - Best model weights
- `card_classifier/run1/results.png` - Training curves
- `card_classifier/run1/confusion_matrix.png` - Confusion matrix

Save `best.pt` to your project: `models/card_hand_classifier.pt`

---

## Step 4: Integrate into Your Code

### 4.1: Update CardHandDetector

Replace template matching with YOLOv8 classification:

```python
from ultralytics import YOLO

class CardHandDetector:
    def __init__(self, classifier_path: str = "models/card_hand_classifier.pt"):
        self.classifier = YOLO(classifier_path)

        # Card slot coordinates (unchanged)
        self.CARD_SLOTS = [
            (156, 1040, 292, 1225),  # Slot 0
            (292, 1040, 427, 1225),  # Slot 1
            (427, 1040, 562, 1225),  # Slot 2
            (562, 1040, 698, 1225),  # Slot 3
        ]

    def identify_card(self, card_img: np.ndarray, threshold: float = 0.7):
        """Identify card using YOLOv8 classifier"""
        if card_img is None or card_img.size == 0:
            return None

        # Run classifier
        results = self.classifier(card_img, verbose=False)

        if len(results) > 0:
            probs = results[0].probs
            card_name = self.classifier.names[int(probs.top1)]
            confidence = float(probs.top1conf)

            if confidence >= threshold:
                return {
                    'card_name': card_name,
                    'card_type': get_card_category(card_name),
                    'confidence': confidence,
                    'available': True  # TODO: Detect grayed out cards
                }

        return None

    def get_hand(self, screenshot: np.ndarray, verbose: bool = False):
        """Detect all 4 cards in hand"""
        hand = []

        for slot_idx, (x1, y1, x2, y2) in enumerate(self.CARD_SLOTS):
            # Crop card from screenshot
            card_img = screenshot[y1:y2, x1:x2]

            # Identify card using YOLO classifier
            card_info = self.identify_card(card_img)
            hand.append(card_info)

        if verbose:
            self._print_hand(hand)

        return hand
```

### 4.2: Detect Grayed Out Cards

To detect if a card is available (not grayed out), add a simple color check:

```python
def _is_card_available(self, card_img: np.ndarray) -> bool:
    """Check if card is available (not grayed out) using color saturation"""
    # Convert to HSV
    hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)

    # Grayed out cards have very low saturation
    avg_saturation = np.mean(hsv[:, :, 1])

    # Available cards have saturation > 50, grayed cards < 30
    return avg_saturation > 40
```

Update `identify_card()`:

```python
def identify_card(self, card_img: np.ndarray, threshold: float = 0.7):
    if card_img is None or card_img.size == 0:
        return None

    # Run classifier
    results = self.classifier(card_img, verbose=False)

    if len(results) > 0:
        probs = results[0].probs
        card_name = self.classifier.names[int(probs.top1)]
        confidence = float(probs.top1conf)

        if confidence >= threshold:
            # Check if available
            is_available = self._is_card_available(card_img)

            return {
                'card_name': card_name,
                'card_type': get_card_category(card_name),
                'confidence': confidence,
                'available': is_available
            }

    return None
```

---

## Step 5: Test and Evaluate

### Test on New Screenshots

```python
from detection.card_hand_detector import CardHandDetector
import cv2

detector = CardHandDetector(classifier_path='models/card_hand_classifier.pt')

# Test on screenshot
screenshot = cv2.imread('test_screenshot.png')
hand = detector.get_hand(screenshot, verbose=True)

print("\nDetected Hand:")
for i, card in enumerate(hand):
    if card:
        print(f"Slot {i}: {card['card_name']} ({card['confidence']:.2f})")
    else:
        print(f"Slot {i}: Unknown")
```

### Run Evaluation Script

```bash
python3 scripts/evaluate_card_classifier.py
```

This will test the classifier on all test images and report:
- Overall accuracy
- Per-card accuracy
- Confusion matrix
- Misclassified examples

---

## Troubleshooting

### Low Accuracy (<90%)

1. **Collect more training data** (50+ images per card)
2. **Use a larger model** (yolov8s-cls or yolov8m-cls instead of yolov8n-cls)
3. **Train for more epochs** (100 instead of 50)
4. **Check for mislabeled images** in your dataset

### Card Confusion (Similar Cards Mixed Up)

1. **Collect more diverse examples** of confused cards
2. **Use data augmentation** (add `augment=True` to training)
3. **Increase model size** (use yolov8m-cls)

### Slow Inference (>20ms per card)

1. **Use smaller model** (yolov8n-cls)
2. **Export to ONNX** for faster inference:
```python
model.export(format='onnx')
# Then load with: YOLO('model.onnx')
```
3. **Reduce image size** to 128x128 or 160x160

### Out of Memory During Training

1. **Reduce batch size** from 32 to 16 or 8
2. **Use smaller model** (yolov8n-cls instead of yolov8m-cls)
3. **Reduce image size** from 224 to 160

---

## Advanced: Fine-Tuning Tips

### Data Augmentation

YOLOv8 automatically applies augmentation. To customize:

```python
results = model.train(
    data='/content/card_hand_classifier',
    epochs=50,
    imgsz=224,
    hsv_h=0.015,    # Hue augmentation
    hsv_s=0.7,      # Saturation augmentation
    hsv_v=0.4,      # Value/brightness augmentation
    degrees=5,      # Rotation (±5 degrees)
    translate=0.1,  # Translation
    scale=0.2,      # Scaling
    flipud=0.0,     # No vertical flip
    fliplr=0.5,     # 50% horizontal flip
)
```

### Transfer Learning from Existing Model

If you already have a card classifier, continue training:

```python
model = YOLO('models/old_card_classifier.pt')  # Load existing model
results = model.train(
    data='/content/card_hand_classifier',
    epochs=25,      # Fewer epochs since starting from trained model
    resume=True     # Resume from last checkpoint
)
```

### Hyperparameter Tuning

Let YOLOv8 find optimal hyperparameters:

```python
model.tune(
    data='/content/card_hand_classifier',
    epochs=50,
    iterations=30,  # Number of tuning iterations
    plots=True
)
```

---

## Performance Comparison

| Method | Accuracy | Speed (per card) | Training Time | Maintenance |
|--------|----------|------------------|---------------|-------------|
| Template Matching | 86.7% | 5ms | None | High (manual templates) |
| YOLOv8n-cls | 95%+ | 15ms | 10 min | Low (auto-learns) |
| YOLOv8s-cls | 97%+ | 20ms | 15 min | Low |
| YOLOv8m-cls | 98%+ | 30ms | 25 min | Low |

**Recommendation**: Start with **YOLOv8n-cls** (nano). If accuracy isn't sufficient, upgrade to yolov8s-cls (small).

---

## Next Steps

1. **Collect training data** using one of the 3 methods above
2. **Train on Google Colab** (free GPU, fastest)
3. **Download trained model** to `models/card_hand_classifier.pt`
4. **Update CardHandDetector** to use YOLOv8 instead of template matching
5. **Test and evaluate** on real gameplay screenshots

Once you have 50+ images per card, training takes only ~10 minutes on Google Colab!
