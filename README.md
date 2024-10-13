# EXPERIMENT N0: 6
# Implementation-of-Logistic-Regression-Using-Gradient-Descent
### NAME : VIDHIYA LAKSHMI S
### REG NO: 212223230238
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the necessary python packages
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value
## Program & Output:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VIDHIYA LAKSHMI S
RegisterNumber:  212223230238
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset = pd.read_csv('Placement_Data_Full_Class (1).csv')
dataset
```
![image](https://github.com/user-attachments/assets/bddcac13-8d1b-4923-b675-b2bb8f6367f5)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/7150dcc0-3327-4e53-a277-abeaa862a81e)
```
dataset.drop('sl_no',axis=1)
dataset.info()
```
![image](https://github.com/user-attachments/assets/b15ee8f8-6797-4c94-8d2e-70e9eb421ab4)
```
dataset["gender"]= dataset["gender"].astype('category')
dataset["ssc_b"]= dataset["ssc_b"].astype('category')
dataset["hsc_b"]= dataset["hsc_b"].astype('category')
dataset["hsc_s"]= dataset["hsc_s"].astype('category')
dataset["degree_t"]= dataset["degree_t"].astype('category')
dataset["workex"]= dataset["workex"].astype('category')
dataset["specialisation"]= dataset["specialisation"].astype('category')
dataset["status"]= dataset["status"].astype('category')
dataset.dtypes
dataset.info()
```
![image](https://github.com/user-attachments/assets/0083473d-e62b-4ccb-81cd-418f172f0616)
```
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/9d0b9898-f33d-4e8e-97bb-8e94b8691e1b)
```
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y
```
![image](https://github.com/user-attachments/assets/861365d1-fb90-4adc-ab6a-32c493fb8bdf)
```
theta = np.random.randn(x.shape[1])
Y=y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,x,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta -= alpha * gradient
    return theta

theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)

def predit(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred=predit(theta,x)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```
![image](https://github.com/user-attachments/assets/e67aee6e-5256-4c87-bd32-4d93ce99ed0f)
```
print(y_pred)
```
![image](https://github.com/user-attachments/assets/cd396b63-9500-4da8-8c65-fb5acde10520)
```
print(y)
```
![image](https://github.com/user-attachments/assets/2484a19d-23f4-4fef-89aa-d300a297171a)
```
xnew = np.array([0,87,0,95,0,2,78,2,0,0,1,0])
y_prednew=predit(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/881da60a-2e98-4446-8ca7-23f25f745513)

```
xnew = np.array([0,0,0,0,0,2,0,2,0,0,1,0])
y_prednew=predit(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/1e635425-26c6-40dc-97f9-1c374a0488c5)

```
print(theta)
```
![image](https://github.com/user-attachments/assets/0e0260f0-bf41-4c17-a099-a9401c96c9ca)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

