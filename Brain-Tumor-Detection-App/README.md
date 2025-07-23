# 🧠 Brain Tumor Detection System
A deep learning-based medical imaging web application that analyzes MRI scans and detects brain tumors with high confidence. Built using TensorFlow, MobileNet, Flask, and OpenCV.

## 📁 Dataset

Located under `Brain-MRI/`, the dataset consists of:

* `yes/` → Tumor-positive MRI scans
* `no/` → Tumor-negative MRI scans


---

## 🧪 Model Accuracy Workflow

1. Auto-validates on boot using sample test cases
2. Applies CLAHE + normalization for input consistency
3. Flags and logs incorrect/inverted predictions
4. Can be re-trained or improved with a custom dataset

---

## 🛠️ Installation & Setup

```bash
# 1. Clone the repository
git https://github.com/mdshoaib9900/Brain-Tumor-Detection-App.git
cd brain-tumor-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```

Open browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🖼️ How to Use

1. Upload an MRI scan image (drag & drop or browse)
2. Click “Start Analysis”
3. View prediction and confidence score

---
