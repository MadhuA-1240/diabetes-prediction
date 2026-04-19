import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    "Glucose": [120, 140, 130, 150, 110],
    "BMI": [25, 30, 28, 32, 24],
    "Diabetes": [0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[["Glucose", "BMI"]]
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict([[135, 29]])
print("Prediction:", prediction[0])
