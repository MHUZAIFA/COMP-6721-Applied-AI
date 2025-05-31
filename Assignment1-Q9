# XGBoost Classifier on Wallet Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


## Step 2: Load Dataset
df = pd.read_excel("R:\Ph.D\COMP6721-AI\Assignments\COMP6721_Assignment1_S25\Wallet.xlsx")
print(df.head())

## Step 3: Preprocess Data
X = df.drop("wallet", axis=1)
y = df["wallet"] - 1  # convert to 0,1,2


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6721)



## Step 4: Train XGBoost Classifier
model = XGBClassifier(
    learning_rate=0.1,
    max_depth=2,
    n_estimators=100,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False
)

model.fit(X_train, y_train)


## Step 5: Evaluate Model
def evaluate_model(y_true, y_pred, dataset_name):
    print(f"--- {dataset_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

evaluate_model(y_train, train_pred, "Training Set")
evaluate_model(y_test, test_pred, "Test Set")


## Step 6: Analyze Learning Rate and Max Depth
results = []
for lr in [0.1, 0.5]:
    for depth in [2, 5]:
        clf = xgb.XGBClassifier(learning_rate=lr, max_depth=depth, n_estimators=100,
                                objective="multi:softmax", num_class=4, eval_metric="mlogloss")
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro')
        results.append((lr, depth, acc, f1))

results_df = pd.DataFrame(results, columns=["Learning Rate", "Max Depth", "Test Accuracy", "F1 Score"])
print("\nComparison of Different Learning Rates and Tree Depths:")
print(results_df)


## Step 7: Visualize Results
sns.barplot(data=results_df, x="Learning Rate", y="Test Accuracy", hue="Max Depth")
plt.title("Test Accuracy by Learning Rate and Max Depth")
plt.show()

sns.barplot(data=results_df, x="Learning Rate", y="F1 Score", hue="Max Depth")
plt.title("F1 Score by Learning Rate and Max Depth")
plt.show()
