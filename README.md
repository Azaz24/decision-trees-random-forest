# Decision Trees vs Random Forest - Customer Churn Analysis

## 📌 Overview
This project compares Decision Tree and Random Forest classifiers on a customer churn dataset to understand overfitting and model performance.

## 🎯 Objectives
- Analyze overfitting in Decision Trees
- Apply hyperparameter tuning (max_depth)
- Compare Decision Tree vs Random Forest
- Select the best model based on performance

## ⚙️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn

## 📊 Results

| Model | Train Accuracy | Test Accuracy |
|------|---------------|--------------|
| Decision Tree (Full) | 1.00 | 0.867 |
| Decision Tree (Best Depth=2) | 0.879 | 0.867 |
| Random Forest | 1.00 | 0.917 |

## 🧠 Key Learnings
- Decision Trees tend to overfit (high variance)
- Controlling depth improves generalisation
- Random Forest reduces overfitting using multiple trees
- Best model selection is based on test performance

## ✅ Conclusion
Random Forest performed best due to higher test accuracy and better generalisation.

## ▶️ How to Run

```
pip install -r requirements.txt
python assignment.py

```

```
📁 Project Structure
decision-trees-random-forest/
│
├── assignment.py
├── README.md
└── requirements.txt
```

## 📌 Dataset
This dataset is synthetically generated using NumPy to simulate customer churn behavior based on features like tenure, monthly charges, services, and support calls.

## 🚀 Future Improvements
- Add cross-validation for better evaluation
- Try other models (e.g., Logistic Regression, XGBoost)
- Perform feature importance analysis

