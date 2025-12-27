# Quick Start: Train Card Hand Classifier

This is the fastest way to get started with YOLOv8 card hand classification.

---

## Option 1: Use Your Existing Templates (5 minutes)

You already have card templates. Convert them to a dataset and train immediately!

### Step 1: Convert Templates to Dataset

```bash
python3 scripts/convert_templates_to_dataset.py
```

This creates `datasets/card_hand_classifier/` with your existing 51 template images split into train/val/test.

### Step 2: Zip Dataset

```bash
cd datasets
zip -r card_hand_classifier.zip card_hand_classifier/
```

### Step 3: Upload to Google Drive

1. Upload `card_hand_classifier.zip` to your Google Drive

### Step 4: Open Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload the notebook: `scripts/train_card_classifier_colab.ipynb`
3. Or create a new notebook and copy the code from [TRAIN_CARD_CLASSIFIER.md](TRAIN_CARD_CLASSIFIER.md)

### Step 5: Train (10 minutes)

Run all cells in the Colab notebook. Training takes ~10 minutes on free T4 GPU.

### Step 6: Download Model

Download `best.pt` and save to `models/card_hand_classifier.pt`

### Step 7: Test

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/card_hand_classifier.pt')

# Test on a card image
img = cv2.imread('test_card.png')
results = model(img)

# Get prediction
probs = results[0].probs
card_name = model.names[int(probs.top1)]
confidence = float(probs.top1conf)

print(f"Detected: {card_name} ({confidence:.2f})")
```

**Done! ✅**

---

## Option 2: Collect More Data (Recommended for Best Accuracy)

Your current templates give you ~51 images across 19 cards (avg 2-3 per card). For better accuracy, collect more data:

### Collect 50+ Images per Card

**Method 1: Auto-collect during gameplay**

```bash
python3 main.py --screenshots --games 20
```

Card crops are saved to `training_data/session_*/card_crops/`

**Method 2: Manual screenshot collection**

1. Play the game and take screenshots
2. Extract cards using:

```python
from detection.card_hand_detector import CardHandDetector
import cv2

detector = CardHandDetector()
screenshot = cv2.imread('screenshot.png')
detector.save_card_crops(screenshot, 'manual_crops')
```

### Organize Into Dataset

```bash
# Put all images into raw_cards/<card_name>/ folders
# Then split into train/val/test:

python3 scripts/split_dataset.py datasets/raw_cards datasets/card_hand_classifier
```

### Train on Larger Dataset

Same as Option 1, but with better accuracy!

---

## Current Status

✅ **Templates converted**: 51 images across 19 cards
✅ **Dataset ready**: `datasets/card_hand_classifier/`
⏭️ **Next**: Upload to Google Drive and train on Colab

---

## Expected Results

With current templates (51 images):
- **Accuracy**: ~85-90% (limited data)
- **Training time**: 10 minutes

With 50+ images per card:
- **Accuracy**: 95-98%
- **Training time**: 15 minutes

---

## Integration After Training

Once you have `models/card_hand_classifier.pt`, see [TRAIN_CARD_CLASSIFIER.md](TRAIN_CARD_CLASSIFIER.md) Step 4 for integration instructions.

Quick preview:

```python
from ultralytics import YOLO
from detection.card_info import get_card_category

class CardHandDetector:
    def __init__(self, classifier_path="models/card_hand_classifier.pt"):
        self.classifier = YOLO(classifier_path)
        # ... rest of init

    def identify_card(self, card_img, threshold=0.7):
        results = self.classifier(card_img, verbose=False)
        probs = results[0].probs

        card_name = self.classifier.names[int(probs.top1)]
        confidence = float(probs.top1conf)

        if confidence >= threshold:
            return {
                'card_name': card_name,
                'card_type': get_card_category(card_name),
                'confidence': confidence
            }
        return None
```

That's it! 🎉
