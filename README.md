# FruitVeg Vision

> Real-time fruit and vegetable recognition from a live webcam feed
> using MobileNetV2 pretrained on ImageNet. Identifies 10 common produce
> categories with confidence scores overlaid directly on the video stream.

## Supported Categories

| Produce     | Produce    |
|-------------|------------|
| Banana      | Tomato     |
| Orange      | Pineapple  |
| Apple       | Broccoli   |
| Cucumber    | Cabbage    |
| Bell Pepper | Carrot     |

## How It Works

```
Webcam Frame (640×480)
        │
        ▼
  Resize to 224×224 + BGR → RGB
        │
        ▼
  MobileNetV2 preprocess_input
        │
        ▼
  ImageNet Prediction (top-5)
        │
        ▼
  Category Mapping Filter
        │
        ▼
  Overlay label + confidence on frame
```

## Installation

```bash
git clone https://github.com/yourusername/fruitveg-vision.git
cd fruitveg-vision
pip install tensorflow opencv-python numpy
```

## Usage

```bash
python fruitveg.py
```

- Point your webcam at a fruit or vegetable
- Recognized item + confidence score appears in green on screen
- Press **`q`** to quit

## Tech Stack

- **MobileNetV2** — Lightweight ImageNet-pretrained CNN via Keras
- **OpenCV** — Webcam capture and frame annotation
- **TensorFlow / Keras** — Model inference and preprocessing
- **NumPy** — Array operations

## Limitations & Roadmap

- Maps only 10 produce categories from ImageNet labels
- Planned: fine-tune on Fruits-360 dataset, add bounding boxes via YOLO, expand category coverage

## Requirements

- Python 3.8+, webcam
- `tensorflow >= 2.x`, `opencv-python`, `numpy`

## License

MIT License
