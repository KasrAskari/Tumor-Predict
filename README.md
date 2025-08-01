# 🧠 Tumor Malignancy Predictor

A simple yet powerful web application that predicts the likelihood of a tumor being malignant or benign using a logistic regression model trained on selected features from a breast cancer dataset.

## 💻 Overview

This is a Streamlit application where you can input tumor characteristics such as radius, perimeter, area, compactness and concavity(Consider reading here for more information) to obtain a probability score on whether the tumor is malignant. The model is trained with logistic regression with hyper-parameter tuning on GridSearchCV.

## 🔍 Features

* Clean UI with interactive inputs powered by Streamlit
* Logistic regression with 10-fold cross-validation
* Hyperparameter tuning using GridSearchCV
* Probability-based prediction with visual feedback
* Built-in data preprocessing (feature scaling)

## 🏗️ Project Structure

```
├── tumor-app.py          # Streamlit application script
├── data.csv              # Dataset with tumor records
├── README.md             # Project documentation
```

## 🛠️ Technologies Used

* **Python 3**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Streamlit**

## 📈 Results

* **Model:** Logistic Regression
* **Best Hyperparameter (C):** Determined via GridSearchCV
* **Cross-Validated Accuracy:** Displayed in-app after training

## 📂 Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Data Set**, publicly available on Kaggle.
🔗 [Click here to download the dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download)

Make sure to download and rename the dataset to `data.csv`, and place it in the same directory as the app script.

Expected columns used in this project:

```
radius_mean, perimeter_mean, area_mean, compactness_mean, concavity_mean, diagnosis
```

## ▶️ How to Run

1. **Install Dependencies** (preferably inside a virtual environment):

```bash
pip install streamlit pandas numpy scikit-learn
```

2. **Download the Dataset** from the [Kaggle link above](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download)
   and save it as `data.csv`.

3. **Run the App:**

```bash
streamlit run tumor-app.py
```

## 📜 License


This project is licensed under the MIT License.
