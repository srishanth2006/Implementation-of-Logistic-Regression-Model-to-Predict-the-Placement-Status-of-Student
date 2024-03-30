# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries such as pandas module to read the corresponding csv file.
2.Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the corresponding dataset values.
4.Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y and Predict the values of array using the variable y_pred.
5.Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
6.Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()


data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or column
data1.head()


data1.isnull().sum()


data1.duplicated().sum()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x


y=data1["status"]
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 0)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")# a library for large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
*/
```

## Output:
![Screenshot 2024-03-30 084342](https://github.com/srishanth2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150319470/2d0376d5-210e-4d8b-8d5c-d8387648c9ea)
![Screenshot 2024-03-30 084356](https://github.com/srishanth2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150319470/3152e2ee-2cc2-4303-b25b-56a2837abc24)
![Screenshot 2024-03-30 084408](https://github.com/srishanth2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150319470/9c29ffbe-c706-4291-86ae-bf720945a78e)
![Screenshot 2024-03-30 084431](https://github.com/srishanth2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150319470/ee18a619-18bc-4857-9ff4-457ab90665c4)
![Screenshot 2024-03-30 084629](https://github.com/srishanth2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150319470/28d9caca-4b6e-4aac-9831-350ccb5a3dff)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
