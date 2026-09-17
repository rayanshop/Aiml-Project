import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("=" * 50)
print("    STUDENT PERFORMANCE PREDICTOR")
print("=" * 50)

data = {"hours": [1, 2, 3, 4, 5, 6, 7, 8], "marks": [35, 40, 50, 55, 65, 70, 80, 90]}

df = pd.DataFrame(data)

X = df[["hours"]]
y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nEnter your study details ")
hours = float(input("  Study Hours: "))

hours_array = np.array([[hours]])

predicted_marks = model.predict(hours_array)

print("\n" + "=" * 50)
print(" RESULT")
print("=" * 50)
print(f" Study Hours   : {hours}")
print(f" Predicted Marks: {int(predicted_marks[0])}")
print("=" * 50)

print("\n Tip: Consistent study = better results!")
