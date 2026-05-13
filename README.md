# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries such as pandas, numpy, matplotlib, and scikit-learn.

2.Load the dataset containing student study hours and corresponding scores using read_csv().

3.Display the first and last few records of the dataset to understand its structure.

4.Separate the independent variable (X) as study hours and dependent variable (Y) as scores.

5.Split the dataset into training and testing sets using the train_test_split() method, with one-third of the data used for testing.

6.Create a Linear Regression model using the LinearRegression() class.

7.Train the regression model using the training dataset (X_train, Y_train).

8.Predict the output values for the testing dataset using the trained model.

9.Compare the predicted values with the actual test values.

Plot the training set results by displaying the scatter plot of actual values and the regression line.

10.Plot the testing set results using the same regression line for comparison.

11.Calculate error metrics such as:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

12.Display the error values to evaluate model performance.



## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: A.HALIMA HARISHA 
RegisterNumber: 212224040094 
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("Last 5 rows of the dataset:")
print(df.tail())

X = df.iloc[:, :-1].values  # Assuming the 'Hours' column is the first column
Y = df.iloc[:, 1].values    # Assuming the 'Scores' column is the second column

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

print("Predicted values:")
print(Y_pred)
print("Actual values:")
print(Y_test)

plt.scatter(X_train, Y_train, color="red", label="Actual Scores")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Fitted Line")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()

plt.scatter(X_test, Y_test, color='green', label="Actual Scores")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print('Mean Squared Error (MSE) =', mse)
print('Mean Absolute Error (MAE) =', mae)
print('Root Mean Squared Error (RMSE) =', rmse)


```

## Output:

<img width="970" height="523" alt="image" src="https://github.com/user-attachments/assets/d486a302-eb3c-4bbe-842f-211ee66a4323" />

<img width="910" height="667" alt="image" src="https://github.com/user-attachments/assets/cab662d6-5264-4b04-a5e3-d5ae8d3acafb" />

<img width="906" height="787" alt="image" src="https://github.com/user-attachments/assets/9f5ddde0-6cf1-4de1-aff2-352f479b8782" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
