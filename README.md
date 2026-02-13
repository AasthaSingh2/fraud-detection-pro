# 💳 Fraud Detection Pro (Streamlit + ML + SHAP)

An end-to-end credit card fraud detection web app built using **Machine Learning** and **Streamlit**, featuring probability-based risk scoring, threshold tuning, and SHAP explainability.

---

## 🔥 Key Features

- ✅ Real-time fraud probability prediction  
- 🎚 Adjustable fraud threshold (precision/recall trade-off)  
- 🧠 SHAP explainability for model decisions  
- 📊 Confusion matrix + ROC curve visualization  
- 💰 Business cost analysis  
- 📥 Downloadable prediction report  

---

## 🛠 Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn (RandomForest)  
- Streamlit  
- SHAP  
- Matplotlib & Seaborn  

---

## 📂 Project Structure

```

fraud-detection/
│
├── app.py
├── fraud_model.pkl
├── X_train.pkl
├── X_test.pkl
├── y_test.pkl
├── requirements.txt
└── README.md

````

---

## ▶️ Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AasthaSingh2/fraud-detection-pro.git
cd fraud-detection-pro
````

### 2️⃣ Create environment (recommended)

```bash
conda create -n fraud_env python=3.10 -y
conda activate fraud_env
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📊 Model Summary

* Model: RandomForestClassifier (class_weight='balanced')
* Evaluation: ROC curve, confusion matrix, classification report
* Dataset is highly imbalanced → threshold tuning is critical

---

## 📁 Dataset

Credit Card Fraud Detection dataset (Kaggle):

Search:

```
creditcard fraud dataset mlg-ulb
```

Dataset is large and not included in this repository.

---
## Screenshots
shap explanbility 
<p align="center">
  <img src="screenshots/shap.png" width="45%" />
</p>




## 🚀 Future Improvements

* Batch CSV fraud prediction
* XGBoost / LightGBM model comparison
* SHAP multi-sample visualization
* Docker deployment

---

## 👩‍💻 Author

**Aastha Singh**
GitHub: [https://github.com/AasthaSingh2](https://github.com/AasthaSingh2)
