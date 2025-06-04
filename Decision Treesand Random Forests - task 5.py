# AI & ML Internship - Task 5: Decision Trees and Random Forests

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "https://raw.githubusercontent.com/rahulrajpl/heart-disease-prediction/master/heart.csv"
df = pd.read_csv(url)

# Check data
print("First 5 rows of data:")
print(df.head())
print("\nData Info:")
print(df.info())

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.title("Decision Tree")
plt.show()

# Analyze overfitting
train_acc = dtree.score(X_train, y_train)
test_acc = dtree.score(X_test, y_test)
print(f"\nDecision Tree Training Accuracy: {train_acc:.2f}")
print(f"Decision Tree Testing Accuracy: {test_acc:.2f}")

# Train Random Forest Classifier
rforest = RandomForestClassifier(n_estimators=100, random_state=42)
rforest.fit(X_train, y_train)

# Predict and evaluate
y_pred = rforest.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
importances = rforest.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Feature Importances from Random Forest")
plt.show()

# Cross-validation
cv_scores = cross_val_score(rforest, X, y, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())