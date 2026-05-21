# 🦾 Ironman — AR Arc Reactor & Repulsor

> Real-time augmented reality that overlays an Iron Man Arc Reactor on your chest and Repulsor beams on your palms — powered by MediaPipe and OpenCV.

---

## 🧠 About

A computer vision AR project that detects your body and hands in real time using a webcam, then composites Iron Man overlays directly onto the live video feed. The Arc Reactor scales with shoulder width; the Repulsor beam scales with palm size — both track your movement frame by frame.

---

## ✨ Features

- **Arc Reactor** — overlaid on chest, scales dynamically with detected shoulder distance
- **Repulsor beam** — overlaid on each detected palm, scales with hand size
- **Multi-hand support** — up to 2 hands tracked simultaneously
- **Transparent PNG compositing** — RGBA overlays preserve alpha channel perfectly
- **Real-time rendering** — runs on standard webcam, no GPU required

---

## 🗂️ Project Structure

```
Ironman/
├── im.py              # Main AR script — pose + hand detection + overlay rendering
├── beam.png           # Repulsor beam image (RGBA)
└── iron-man-arc.png   # Arc Reactor image (RGBA)
```

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.8+
- Webcam
- `beam.png` and `iron-man-arc.png` in the same directory as `im.py`

### Install dependencies

```bash
pip install opencv-python mediapipe pillow numpy
```

### Run

```bash
python im.py
```

Press `ESC` to exit.

---

## 🔧 How It Works

**Pose detection (`MediaPipe Pose`)**
- Detects left and right shoulder landmarks each frame
- Computes shoulder pixel distance to dynamically size the Arc Reactor
- Places the reactor centered between shoulders, offset downward to chest position

**Hand detection (`MediaPipe Hands`)**
- Detects up to 2 hands per frame
- Uses INDEX_FINGER_MCP and PINKY_MCP landmarks to determine palm center and width
- Scales Repulsor size proportionally to palm width

**Overlay compositing**
- Uses PIL (`Image.paste` with alpha mask) for clean RGBA transparency blending
- Converts between BGR (OpenCV) and RGB (PIL) each frame
- Handles edge-case failures gracefully with try/except per overlay

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Computer Vision | OpenCV (`cv2`) |
| Body & Hand Tracking | MediaPipe Pose + Hands |
| Image Compositing | Pillow (PIL) |
| Numerical ops | NumPy |

---

## 📄 License

Open source — free to use and modify.

---

*Built by [Pravin Kumar M](https://github.com/Pravinpk26)*
