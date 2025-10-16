# ♻️ Real-Time Scrap Classifier & Robotic Pick Simulation

This project simulates an **AI-powered scrap sorting system** that uses **YOLOv8**, **OpenCV**, and **Streamlit** to detect recyclable materials in real time and generate robotic pick points.

---

## 🎯 Objective
The goal is to classify waste/scrap materials such as **metal, plastic, paper, cardboard, glass, and trash** using a trained deep learning model and visualize results through a live dashboard.

---

## 🧩 Features
✅ Real-time object detection using YOLOv8  
✅ Classifies 6 waste materials  
✅ Pick-point generation (center of detected object)  
✅ Streamlit dashboard with:
- Live webcam feed  
- Object counts per class  
- Detection timestamps  
- Real-time bar chart updates  

✅ CSV logging of detections  
✅ Ready for integration with robotic arms (simulation)

---

## 🧱 Project Structure
ScrapClassifier/
│
├── convert_to_yolo.py # Converts dataset to YOLO format
├── realtime_infer.py # OpenCV-based live detection
├── realtime_dashboard.py # Streamlit dashboard UI
├── dataset/ # YOLO-ready dataset
├── runs/detect/train2/weights/ # Trained YOLO model
│ └── best.pt
├── outputs/ # CSV and logs
│ └── picks.csv
├── data_raw/ # Original raw dataset
├── requirements.txt # Required Python packages
└── README.md # Project documentation



---

## ⚙️ Tech Stack
| Tool / Library | Purpose |
|----------------|----------|
| **Python 3.10+** | Programming language |
| **YOLOv8 (Ultralytics)** | Object detection |
| **OpenCV** | Real-time webcam input & image processing |
| **Streamlit** | Interactive dashboard |
| **Pandas / NumPy** | Data processing |
| **Matplotlib** | Visualization |

---

## 🧠 Dataset
**Waste Classification Data** from [Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)

Classes used:
- Cardboard  
- Glass  
- Metal  
- Paper  
- Plastic  
- Trash  

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Suryawanshitejas158/ScrapClassifier.git
cd ScrapClassifier

2️⃣ Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Run the YOLO Training
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=20 imgsz=640

5️⃣ Run Real-Time Detection (OpenCV)
python realtime_infer.py

6️⃣ Run Streamlit Dashboard
streamlit run realtime_dashboard.py


🧩 Streamlit Dashboard Includes:

📸 Live Webcam Feed
🧱 Material Class (Paper, Plastic, etc.)
🔢 Object Count
🕒 Timestamp
🧾 Output Example
📊 Live Bar Chart:

🧾 Output Example

| Timestamp           | Material | Confidence |
| ------------------- | -------- | ---------- |
| 2025-10-14 21:02:17 | plastic  | 0.92       |
| 2025-10-14 21:02:19 | paper    | 0.88       |



📂 Results

Training Accuracy: 98.8% mAP@50

Model: yolov8n.pt (fine-tuned on Waste Dataset)

Inference Speed: ~80ms per frame on CPU




## 🎥 Demo Video

Here’s the real-time working of my Scrap Classifier 👇  
https://github.com/user-attachments/assets/088f8501-0811-4e78-8e9b-91c5d0852cca


👨‍💻 Author
Name: Tejas Suryawanshi
Role: Computer Vision Engineer Intern
GitHub: Suryawanshitejas158
Email: (tejassuryawanshi7249@gmail.com)
