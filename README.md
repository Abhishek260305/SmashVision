## 📝 Introduction

SmashVision is an **AI-powered badminton score detection system** that acts like a real umpire. It automatically detects which player scores each point in a rally, tracks shuttle and player movements with high accuracy, estimates their speeds, and visualizes player movement directions. Built using YOLOv8, Detectron2, Kalman filters, and Deep SORT, SmashVision handles edge cases such as near-boundary shots, net shots, and shuttle occlusions to provide reliable scoring and analytics for badminton singles matches.

## 🔽 Model Weights

⚠️ Make sure to **download weights manually** and place them in the correct folder before running.

Download the required model weights before running:

- [YOLOv8 Weights (Hugging Face)](https://huggingface.co/ThunderBird325/SmashVision-YOLOv8-1280-Weights/blob/main/best.pt)
- [Court KP-RCNN Weights (Hugging Face)](https://huggingface.co/ThunderBird325/SmashVision-Court-KPRCNN-Weights/blob/main/model_final.pth)
- [Net KP-RCNN Weights (Hugging Face)](https://huggingface.co/ThunderBird325/SmashVision-Net-KPRCNN-Weights/blob/main/net_kprcnn.pth)

🔹 **Optional (Not used currently):**

- [Mask R-CNN Weights (Hugging Face)](https://huggingface.co/ThunderBird325/SmashVision-MaskRCNN-Weights/blob/main/mask_rcnn.pth)

### 📁 Models Folder Structure

Place them in your `models/` folder as:
```
models/
├── best.pt
├── model_final.pth
├── net_kprcnn.pth
└── mask_rcnn.pth # optional
```
## ✨ Features

✅ **Automated Badminton Scoring**  
Detects which player wins each point based on the shuttle landing position and the rally outcome.

✅ **Real-Time Object Detection**  
Uses YOLOv8 for detecting players, shuttle, and rackets with high accuracy.

✅ **Court & Net Detection**  
Extracts court boundaries and net position using KP-RCNN models to determine in/out shots.

✅ **Trajectory Smoothing**  
Implements a Kalman 6D filter for smoother shuttle tracking even under occlusions or motion blur.

✅ **Consistent Player Tracking**  
Uses Deep SORT for unique player IDs throughout the video to maintain scoring accuracy.

✅ **Speed & Movement Estimation**  
Calculates speed of the shuttle and players; displays an arrow indicating player movement direction.

✅ **Handles Edge Cases**  
Robust to near-boundary landings, net shots, obscure shuttle frames, and camera variations.

✅ **Optimized Pipeline**  
Threaded processing for improved frame rates and real-time performance on an RTX 4050 laptop GPU.

✅ **Modular Design**  
Easily extendable to doubles games, live AI commentary, and training insights (footwork, court coverage).

✅ **Large Diverse Dataset**  
Trained on over 50,000+ annotated images collected via Roboflow, CVAT, and Label Studio with extensive augmentation for generalization.

🔹 **Optional:** Mask R-CNN for advanced court/net segmentation (not integrated in current version).

## 💻 Technical Stack

### 🔹 **Frameworks & Libraries**
- **PyTorch** – Core deep learning framework for model training and inference
- **Detectron2** – For KP-RCNN and Mask R-CNN models (court and net detection)
- **Ultralytics YOLOv8** – For shuttle, player, and racket detection
- **OpenCV** – Video processing, drawing overlays, and pipeline integration
- **Deep SORT Realtime** – Player tracking with consistent IDs
- **FilterPy** – Kalman 6D filter implementation for trajectory smoothing
- **Pandas & NumPy** – Data processing and analysis

### 🔹 **Annotation & Dataset Tools**
- **Roboflow** – Dataset management and preprocessing
- **CVAT** – Manual annotation for keypoints and bounding boxes
- **Label Studio** – Additional annotation tool for dataset diversity

### 🔹 **Optimization & Utilities**
- **Threading** – For parallelizing detection and tracking processes
- **Matplotlib** – For debugging visualizations and plots

### 🔹 **Environment**
- **Python 3.10**
- **Laptop GPU:** RTX 4050 (6GB VRAM)
- **Hugging Face Hub** – For model weight hosting and sharing
- **Git & Git LFS** – Version control and large file storage

## 📁 Project Structure
```
SmashVision/
├── court_module.py
├── deep_sort_module.py
├── kalman6D.py
├── main.py
├── net_utils.py
├── player_assignment.py
├── player_detector.py
├── README.md
├── requirements.txt
├── requirements_mac.txt
├── shuttle_utils.py
│
├── Models/
│ ├── COURT(kprcnn)/
│ │ └── model_final.pth
│ ├── Mask(court+net)/
│ │ └── model_final.pth
│ ├── NET(kprcnn)/
│ │ └── model_final.pth
│ └── YOLO(person+shuttle)/
│ └── best.pt
│
├── output/
│ ├── courtDebug.py
│ ├── court_polygon.json
│ ├── net_debug.py
│ └── net_polygon.json
│
├── video_outputs/
│ ├── test1_output.mp4
│ ├── test2_output.mp4
│ ├── test3_output.mp4
│ ├── test5_output.mp4
│ ├── testCase1_output.mp4
│ ├── testCase2_output.mp4
│ ├── testCase3_output.mp4
│ ├── testCase5_output.mp4
│ ├── testCase6_output.mp4
│ └── testCase6_output_with_speed.mp4
```

## 🚀 How to Use

Follow these steps to run **SmashVision** on your local machine:

---

### 🔹 1. Clone the repository

First, clone the GitHub repository and navigate into the project directory:

```bash
git clone https://github.com/Abhishek260305/SmashVision.git
cd SmashVision
```
### 🔹 2. Install the Dependencies

   For Windows
   ```bash
   pip install -r requirements.txt
   ```
   For macOS
   ```bash
   pip install -r requirements_mac.txt
   ```

### 🔹 3. Run SmashVision
   
   For Singles Game
   ```bash
   python main.py
   ```

### 🔹 4. View Outputs

   The output videos will be saved in the video_outputs/ folder with overlays showing:

   Player detection

   Shuttle trajectory

   Speed estimations

   Movement direction arrows

   Scoring results

## 🏋️‍♂️ Model Training

### 🔹 **1. YOLOv8 (Player & Shuttle Detection)**

- **Model:** YOLOv8-Large
- **Image Resolution:** 1280 x 1280
- **Dataset:** Combination of Roboflow dataset and self-collected data  
  [🔗 Roboflow Dataset Link](https://app.roboflow.com/abc-9i5bp/badminton-hehp8-x0mwi/2)
- **Training Tool:** Roboflow for annotation and dataset management, Ultralytics YOLOv8 for training

---

### 🔹 **2. KP-RCNN (Court & Net Detection)**

- **Model:** Keypoint R-CNN (Detectron2)
- **Config Used:** `COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml`
- **Image Resolution:** 640 x 360
- **Dataset:** Self-annotated images using CVAT
- **Annotation Tool:** CVAT running locally via Docker  
---

### 🔹 **3. Mask R-CNN (Optional)**

- **Model:** Mask R-CNN (Detectron2)
- **Config Used:** `COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml`
- **Image Resolution:** 1024 x 1024
- **Dataset:** Self-annotated dataset on Roboflow  
  [🔗 Roboflow Dataset Link](https://app.roboflow.com/abc-9i5bp/badmintonc-btpug/6)

🔴 **Note:** Mask R-CNN is currently **optional** and not integrated in this version.

---

### 🔧 **Training Infrastructure**

- All models were trained and tested on an **RTX 4050 laptop GPU (6GB VRAM)** with optimized batch sizes and augmentations for stability and generalization.

## 🎯 Roadmap

- [ ] Extend to doubles game scoring
- [ ] Integrate AI-generated live commentary
- [ ] Full rally-by-rally match scoring and analytics
- [ ] Deploy as an AI badminton trainer for footwork, court coverage, and tactical feedback

---

### 💡 **Contributions**

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

### 🤝 **Contact**

For any questions or collaborations, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abhishek-singh-92a087247/) or raise an issue.

---
