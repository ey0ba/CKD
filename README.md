# 🧠 Predictive Modeling for Chronic Kidney Disease (CKD)

A machine learning project, demonstrating a full data science workflow for predicting Chronic Kidney Disease (CKD) using clinical data.

This repository includes:
- Data preprocessing & cleaning  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Correlation-based feature selection  
- Model training using a **Random Forest Classifier**  
- Cross‑validation for robustness  
- Feature importance analysis  

---

## 📁 Project Structure

```
CKD/
│── ckd_portfolio.ipynb      # Main analysis notebook
│── README.md                # Project documentation
│── requirements.txt         # Dependencies
│── ckd.csv                  # Input dataset (~400 rows; ~24 raw features)
│── .gitignore               # Git ignore rules
```

---

## 🚀 Project Overview

This project applies supervised machine learning to identify CKD based on routine clinical measurements.  
The dataset contains **~400 rows**, which reduce to **~300+ usable samples** after cleaning.  

There are **~24 original variables**, expanding to **~40+ encoded features** after one‑hot encoding.

---

## 📊 Workflow Summary

### **1. Data Preprocessing**
- Removed invalid `"discrete"` entries  
- Dropped rows with missing values  
- Classified variables into:
  - Binary  
  - Categorical  
  - Numerical  
- Encoded:
  - Binary → `LabelEncoder`  
  - Categorical → One‑Hot Encoding  

---

### **2. Feature Selection**
- Generated a correlation matrix  
- Removed features with correlation **< 0.40**  
- Dropped the `affected` column (perfect correlation → leakage)

---

### **3. Model Development**
- Train/test split: **80% training, 20% testing**
- Model used: **RandomForestClassifier**
- Performance evaluated using accuracy score  
- Feature importance visualized via bar plot  

---

### **4. Cross‑Validation**
Performed **5‑fold cross‑validation**:

| Fold | Accuracy |
|------|----------|
| 1 | 0.95 |
| 2 | 0.95 |
| 3 | 0.925 |
| 4 | 0.975 |
| 5 | 0.95 |

**Mean CV Accuracy:** **0.95 (95%)**

---

## 🏆 Results

### **Model Performance**
- **Random Forest Accuracy:** **95%**
- **Cross‑validation mean accuracy:** **95%**

### **Most Important Predictive Features**
- `al_<0` (albumin)  
- `dm` (diabetes mellitus)  
- `htn` (hypertension)  
- `sg_≥1.023` (specific gravity)  
- `bp limit_*` encoded categories  

These align with known CKD risk factors.

---

## 📘 Interpretation of Key Clinical Variables

| Feature | Meaning |
|--------|---------|
| **al** | Albumin level (indicator of kidney damage) |
| **dm** | Diabetes mellitus (major CKD risk factor) |
| **htn** | Hypertension (common cause of CKD) |
| **sg** | Specific gravity of urine (kidney concentration ability) |
| **bp limit** | Encoded blood pressure categories |

---

## 📦 Installation & Usage

### **1. Clone the repository**
```bash
git clone git@github.com:ey0ba/CKD.git
cd CKD
```

### **2. Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Launch Jupyter Notebook**
```bash
jupyter notebook
```

---

## 👨‍⚕️ Author  
**Eyob Assefa Betiru**  
Machine Learning • Data Science • Health Informatics  

---

## 📄 License  
This project is for educational and research purposes.
