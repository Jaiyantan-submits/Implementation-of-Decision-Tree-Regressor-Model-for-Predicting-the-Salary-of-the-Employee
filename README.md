# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Jaiyantan S
RegisterNumber:  212224100021
*/
```

```
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
x.head()
y = data["Salary"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 score: ", r2)

dt.predict([[5, 6]])

```

## Output:

![image](https://github.com/user-attachments/assets/4f585784-9d3e-4718-8682-9e01fd811d41)

![image](https://github.com/user-attachments/assets/19c3ea24-d048-40ed-ae94-c23cc9d27c7d)

![image](https://github.com/user-attachments/assets/7922a743-819e-4317-adb7-e25610b99c7a)

![image](https://github.com/user-attachments/assets/a2de50dd-2be7-49fa-adf9-292df87d3935)

![image](https://github.com/user-attachments/assets/ecf99853-2f8a-42ac-b5f4-0bc297bf5a83)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
