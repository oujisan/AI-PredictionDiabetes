# 🩺 Optimized Diabetes Risk Prediction using XGBoost

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project aims to build an **Early Warning System** for diabetes risk using machine learning. We utilized the **CDC's BRFSS 2015 Health Indicators Dataset**.

The main challenge of this project was handling **Extreme Class Imbalance** where Healthy patients significantly outnumbered Diabetic patients. Our solution involves a strategic combination of **Binary Classification**, **Random Undersampling (2:1 Ratio)**, and **Feature Selection** using XGBoost to achieve a high Sensitivity (Recall) for medical screening purposes.

👉 **[View Live Demo](#)** *(Add your streamlit share link here if deployed)*

## 📂 Dataset Source
The dataset is obtained from the **Behavioral Risk Factor Surveillance System (BRFSS)** 2015 by the CDC.
- **Original Size:** 253,680 rows, 22 columns.
- **Source:** [Kaggle - Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

## ⚠️ The Challenge: Class Imbalance
Initially, the dataset had 3 classes:
1.  **0 (Healthy):** ~213k samples (Majority)
2.  **1 (Pre-diabetes):** ~4.6k samples (Minority - Hard to detect)
3.  **2 (Diabetes):** ~35k samples

**Problem:** Standard models failed to detect Pre-diabetes (Recall < 5%) due to feature overlap with Healthy patients.

**Our Solution:**
1.  **Binary Transformation:** Merged Class 1 (Pre-diabetes) and Class 2 (Diabetes) into a single **"At Risk"** class.
2.  **Undersampling Strategy:** We reduced the majority class (Healthy) to achieve a **2:1 Ratio** (2 Healthy : 1 At Risk). This improved training speed and model sensitivity without losing critical patterns.

## 🛠️ Methodology (CRISP-DM)

### 1. Data Preprocessing
- **Duplicate Removal:** Removed ~24k duplicate rows.
- **Sanity Check:** Removed impossible biological values (e.g., BMI > 90).
- **Encoding:** Features like Age (1-13) and Income (1-8) are ordinal codes from the CDC questionnaire.

### 2. Feature Selection (Embedded Method)
We used **XGBoost Feature Importance** to select the most relevant health indicators.
- **Initial Features:** 21 Features.
- **Selected Features:** 13 Features (HighBP, BMI, GenHlth, Age, Income, etc.).
- **Discarded Features:** Fruits, Veggies, NoDocbcCost (proven to have low impact on the model).

### 3. Modeling
- **Algorithm:** XGBoost Classifier.
- **Hyperparameters:** `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`.
- **Scaling:** MinMaxScaler.

## 📊 Model Evaluation
We prioritized **Recall (Sensitivity)** to minimize False Negatives (missing a diabetic patient is dangerous).

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **79%** | The model successfully detects 79% of at-risk patients. |
| **Precision** | 34% | Acceptable trade-off for a screening tool (better to be safe). |
| **Accuracy** | 72% | Stable performance on real-world unseen data. |
| **F1-Score** | 0.47 | Balanced metric for the minority class. |

## 💻 Tech Stack
- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Imbalanced-learn (imblearn), XGBoost
- **Deployment:** Streamlit
- **Visualization:** Matplotlib, Seaborn

## 🚀 How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/your-username/diabetes-prediction.git](https://github.com/your-username/diabetes-prediction.git)
   cd diabetes-prediction
````

2.  **Install Requirements**
    Make sure you have Python installed.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**

    ```bash
    streamlit run main.py
    ```

## 📝 Data Dictionary (Why Age is 1-13?)

The input data follows the CDC Codebook standards:

  - **Age (1-13):** 1 = 18-24 years ... 13 = 80+ years.
  - **Education (1-6):** 1 = Kindergarten ... 6 = College Graduate.
  - **Income (1-8):** 1 = \<$10k ... 8 = >$75k.
  - **General Health (1-5):** 1 = Excellent ... 5 = Poor.

-----

*Disclaimer: This tool is for educational and screening purposes only. It does not replace professional medical advice.*

````
