# ================================
# IMPORT LIBRARIES
#=================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# DATASET CREATION
# ==============================
np.random.seed(42)
n = 300

data = {
    'tenure':          np.random.randint(1, 72, n),
    'monthly_charges': np.round(np.random.uniform(20, 120, n), 2),
    'num_services':    np.random.randint(1, 8, n),
    'contract_type':   np.random.randint(0, 3, n),
    'support_calls':   np.random.randint(0, 10, n),
}
df = pd.DataFrame(data)

log_odds = (
    -0.05 * df['tenure']
    + 0.02 * df['monthly_charges']
    - 0.3  * df['num_services']
    - 0.9  * df['contract_type']
    + 0.1  * df['support_calls']
    + np.random.randn(n) * 0.8
)
prob = 1 / (1 + np.exp(-log_odds))
df['churn'] = (prob > 0.55).astype(int)

# ==============================
# FEATURES & TARGET
# ==============================
X = df.drop('churn', axis=1)
y = df['churn']

# ==============================
# TRAIN-TEST SPLIT (80/20)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# TASK 1: DECISION TREE (NO LIMIT)
# ==============================
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
test_acc_full = accuracy_score(y_test, dt_full.predict(X_test))

print("\n--- Task 1: Decision Tree (No Limit) ---")
print("Train Accuracy:", round(train_acc_full, 3))
print("Test Accuracy :", round(test_acc_full, 3))

# COMMENT:
# Agar train accuracy high aur test accuracy low hai → OVERFITTING

# ==============================
# TASK 2: HYPERPARAMETER TUNING
# ==============================
depths = [2, 4, 6, 8, 10]
best_depth = None
best_test_acc = 0

print("\n--- Task 2: Depth Tuning ---")

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Depth {d} → Train: {round(train_acc,3)} | Test: {round(test_acc,3)}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_depth = d
        best_model = model

print("\nBest Depth:", best_depth)

# COMMENT:
# Depth badhne par:
# Bias ↓ (model zyada learn karta hai)
# Variance ↑ (overfitting ka risk badhta hai)

# ==============================
# TASK 3: RANDOM FOREST
# ==============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

train_acc_rf = accuracy_score(y_train, rf.predict(X_train))
test_acc_rf = accuracy_score(y_test, rf.predict(X_test))

print("\n--- Task 3: Random Forest ---")
print("Train Accuracy:", round(train_acc_rf, 3))
print("Test Accuracy :", round(test_acc_rf, 3))

# COMMENT:
# Random Forest multiple trees ka average leta hai
# Isliye overfitting kam hota hai → train-test gap chhota hota hai

# ==============================
# TASK 4: FINAL COMPARISON
# ==============================
print("\n--- Task 4: Model Comparison ---")

print(f"Decision Tree (Full)  → Train: {round(train_acc_full,3)} | Test: {round(test_acc_full,3)}")
print(f"Decision Tree (Best)  → Train: {round(best_model.score(X_train, y_train),3)} | Test: {round(best_test_acc,3)}")
print(f"Random Forest        → Train: {round(train_acc_rf,3)} | Test: {round(test_acc_rf,3)}")

# FINAL COMMENT:
# Best model usually Random Forest hota hai
# Kyunki:
# - Stable performance
# - Overfitting kam
# - Real-world me reliable