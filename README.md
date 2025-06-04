# 🌸 K-Nearest Neighbors (KNN) Classifier – Iris Dataset

## 🎯 Objective
To implement and understand the K-Nearest Neighbors (KNN) algorithm for classifying the famous Iris flower dataset using scikit-learn.

---

## 📚 Dataset Used
*Iris Dataset*  
- 150 samples  
- 4 features:  
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  
- 3 Target Classes:  
  - Setosa  
  - Versicolour  
  - Virginica  

Dataset Source: Loaded directly using sklearn.datasets.load_iris() — No need to download separately.

---

## 🛠 Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔎 Steps Performed
1. Loaded the Iris dataset using sklearn.datasets.
2. Normalized features using StandardScaler.
3. Split the dataset into train and test using train_test_split.
4. Trained the KNN model for K = 1 to 10.
5. Chose the best K based on accuracy score.
6. Evaluated the model using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
7. Visualized:
   - Accuracy vs K (line graph)
   - Decision boundary using 2 selected features

---

## 📈 Output Files
- accuracy_vs_k.png – Accuracy score for different K values  
- decision_boundary.png – Visualization of decision regions (2D)  
- classification_report.txt – Evaluation results  

---

## ▶ How to Run
Make sure Python and the required libraries are installed.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python knn_classifier.py
