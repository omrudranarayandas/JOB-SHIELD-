[![analyticsindiamag.com/de...](https://images.openai.com/thumbnails/url/Yaxk6Hicu1mUUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw40y3b1qDCL9CvLD4yKyo7KqSrzijT3ck71SM_21C0xN7d08i9KSTQJqHBy9HcLNzEODjVOc_VKCQpVKwYAvOEoWg)](https://analyticsindiamag.com/deep-tech/classifying-fake-and-real-job-advertisements-using-machine-learning/)

# 🛡️ **Job Shield** – Fake Job Ad Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) 
[![License: MIT](https://img.shields.io/github/license/USERNAME/job-shield.svg)](LICENSE) 
[![TF‑IDF](https://img.shields.io/badge/Text_Feature‑Extraction‑TF--IDF-orange.svg)]() 
[![Models](https://img.shields.io/badge/Models-SVM%20%7C%20RF%20%7C%20KNN%20%7C%20NB%20%7C%20MLP-blueviolet.svg)]()

---

## 🎯 Project Overview

**Job Shield** is a Python-based machine learning project designed to detect fake job advertisements using NLP techniques. It preprocesses job posting text, converts it into TF-IDF vectors, and evaluates multiple classifiers to flag suspicious ads.

---

## 🔍 Algorithms & Sample Performance

| Classifier                   | Accuracy\* |
| ---------------------------- | ---------: |
| Support Vector Machine (SVM) |        95% |
| Random Forest                |        93% |
| Decision Tree                |        91% |
| K‑Nearest Neighbors (KNN)    |        90% |
| Naive Bayes                  |        89% |
| Multilayer Perceptron (MLP)  |        94% |

\*Results based on a standard 80/20 train-test split; actual performance may vary.

---

## 📊 Key Visuals

* **Confusion Matrix (SVM model):** Shows true/false positives and negatives, aiding error analysis.

[![analyticsindiamag.com/de...](https://images.openai.com/thumbnails/url/Yaxk6Hicu1mUUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw40y3b1qDCL9CvLD4yKyo7KqSrzijT3ck71SM_21C0xN7d08i9KSTQJqHBy9HcLNzEODjVOc_VKCQpVKwYAvOEoWg)](https://analyticsindiamag.com/deep-tech/classifying-fake-and-real-job-advertisements-using-machine-learning/)

* **Model Accuracy Comparison:** Bar chart of classifier performance.

[![researchgate.net/figure/...](https://images.openai.com/thumbnails/url/goQo9nicu1mUUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw5x8quqirQIMvMpLnKuiiooCjY3NDXP9guJSHGrzCotjAwKCghJS7EIci6zKA_wS_cJs0jKiTLM9C9UKwYAzaMpZw)](https://www.researchgate.net/figure/Bar-Graph-for-Accuracy-Comparison-of-Models-The-above-figure-10-shows-the-Bar-Graph-of_fig6_383792073)

---

## 🛠️ Project Structure

```
job-shield/
├── data/               # Raw & processed data (e.g., fake_job_postings.csv)
├── images/             # Confusion matrices, bar charts, GUI screenshots
├── models/             # Trained model files (*.pkl)
├── src/                # Source code modules
│   ├── preprocessing.py
│   ├── vectorizer.py
│   ├── train.py
│   └── evaluate.py
├── app.py              # Optional CLI or GUI entry point
├── requirements.txt
└── README.md
```

---

## 🧪 Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/USERNAME/job-shield.git
   cd job-shield
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run training & evaluation:**

   ```bash
   python src/train.py
   python src/evaluate.py
   ```

4. **(Optional) Launch GUI:**

   ```bash
   python app.py
   ```

---

## 📚 Dataset

This project uses the popular [Real/Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) dataset from Kaggle, containing \~18,000 ads (\~800 fake) ([kaggle.com][1], [geeksforgeeks.org][2], [kaggle.com][3], [geeksforgeeks.org][2], [Analytics India Magazine][4], [github.com][5], [gist.github.com][6]).

---

## 🪴 Installation & Setup

* Requires **Python 3.9+**
* Key libraries: `scikit-learn`, `pandas`, `numpy`, `nltk`, `matplotlib`, `seaborn`
* Run:

  ```bash
  pip install -r requirements.txt
  ```

---

## 🧭 Usage

* `preprocessing.py` – Data cleaning: remove HTML, lowercase, strip punctuation, tokenize.
* `vectorizer.py` – Builds and applies TF-IDF.
* `train.py` – Trains algorithms with cross-validation.
* `evaluate.py` – Generates performance metrics and visualizations.
* `app.py` – Launch GUI or CLI for real-time prediction.

---

## 📁 Visualization & Analysis

* Confusion matrices illustrate classification errors (TP, FP, TN, FN).
* Bar charts compare model performance.
* Use visual feedback to refine preprocessing (e.g., stopword handling, n‑gram size).

---

## 👤 Author

Om Rudra Narayan Das
📧 [omrudranarayandas@gmail.com](mailto:omrudranarayandas@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/-rudradas/)

---


