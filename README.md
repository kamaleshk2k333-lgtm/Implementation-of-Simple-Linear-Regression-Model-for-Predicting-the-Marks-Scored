# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Create a dataset with Hours_Studied as the independent variable and Marks_Scored as the dependent variable, then store it in a pandas DataFrame.
2.Split the dataset into training and testing sets using train_test_split() to evaluate the model’s performance.

3.Train a Linear Regression model using the training data, predict marks for the test data, and evaluate the model using Mean Squared Error and R² score.

4.Plot the regression line with actual data points and use the trained model to predict marks for a given number of study hours. 

## Program:
```

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:kamaleshkumar k
RegisterNumber:25012000
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```
```
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df
```
```
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
```
model = LinearRegression()
model.fit(X_train, y_train)
```
```
y_pred = model.predict(X_test)
```
```
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```
```
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X, model.predict(X), color='black', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
```
```
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
```

## Output:

<img width="539" height="549" alt="540927121-fe001bf2-0f98-48a6-b49f-6d738c4f75c6" src="https://github.com/user-attachments/assets/815f4251-cbd6-4e5d-9df3-29d4eb81f1f8" />
<img width="435" height="172" alt="540927380-1dc260c5-8f51-46c6-b014-79e6d7e54263" src="https://github.com/user-attachments/assets/2ce46c79-3118-4e0d-9d82-8931ea9d2ab9" />
<img width="311" height="79" alt="540927268-86ce3b40-3e65-4170-acbf-6ac1541fd207" src="https://github.com/user-attachments/assets/7d1feeb4-944d-4ff4-b04b-396577fe8791" />
<img width="1018" height="685" alt="540927565-ecabdc83-b8ea-4d8f-8774-1595bbb58fbe" src="https://github.com/user-attachments/assets/a13686f3-76b6-433d-9c13-b56e0741842e" />
<img width="1037" height="102" alt="540927668-ac51d12a-0cbb-4684-bb49-3b8fe42d0f52" src="https://github.com/user-attachments/assets/31455789-e7ee-4bff-8fac-0b278e0dda63" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
