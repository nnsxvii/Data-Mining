import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    'Age': [29, 34, 40, 23, 53, 38, 25, 32, 41, 55],
    'Salary': [32000, 42000, 75000, 27000, 81000, 60000, 30000, 48000, 67000, 85000],
    'Approved': [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]  # 0 = Not Approved, 1 = Approved
}

df = pd.DataFrame(data)

X = df.drop('Approved', axis=1)  # Features: 'Age' and 'Salary'
y = df['Approved']  # Target: 'Approved'

X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X, y, test_size=0.20, random_state=42)

X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, test_size=0.30, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train_80, y_train_80)
predictions_80 = model.predict(X_test_20)
accuracy_80 = accuracy_score(y_test_20, predictions_80)

model.fit(X_train_70, y_train_70)
predictions_70 = model.predict(X_test_30)
accuracy_70 = accuracy_score(y_test_30, predictions_70)

print(f"Accuracy with 80-20 split: {accuracy_80:.2f}")
print(f"Accuracy with 70-30 split: {accuracy_70:.2f}")
