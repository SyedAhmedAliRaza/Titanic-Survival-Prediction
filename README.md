# 🚢 Titanic Survival Prediction  

This project implements a **machine learning pipeline** to predict survival on the Titanic dataset using a **Decision Tree Classifier**. It includes **data preprocessing, feature engineering, model training, evaluation, and visualizations** for better insights into the survival patterns.  

---

## 📌 Project Overview  
The goal of this project is to analyze the Titanic dataset, clean and preprocess the data, engineer new features, and train a Decision Tree model to predict passenger survival. Visualizations are included to explore relationships between survival and key features such as **age, gender, class, family size, and fare**.  

---

## ⚙️ Features  
- Handles missing values (`age`, `embarked`).  
- Feature engineering:
  - Family size (`family`, `alone`).  
  - Age grouping into bins.  
  - Fare grouping using quartiles.  
  - Grouping `sibsp` and `parch`.  
- Label encoding for categorical variables.  
- Trains a **Decision Tree Classifier** with class balancing.  
- Evaluates performance using **accuracy, recall, and confusion matrix**.  
- Provides extensive **data visualizations**:
  - Survival by demographic and travel features.  
  - Count plots and box plots for deeper insights.  

---

## 📂 Dataset  
- The dataset is loaded directly from **Seaborn**:  
  ```python
  data = sns.load_dataset('titanic')
  
## 💻 How to Run 

Use the following link on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TmoA28gNQdz1Ln0Z5isZd-zocc-iMpB4?usp=sharing)
