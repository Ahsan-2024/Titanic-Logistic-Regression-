import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
Data= pd.read_csv("train.csv")
print(Data)
Data.info()
Data.isnull()
Data.isnull().sum()
NewData = Data.drop("Cabin",axis=1)
NewData.info()
Average = NewData["Age"].mean()
NewData["Age"] = NewData["Age"].fillna(Average)
NewData.isnull().sum()
Common = NewData["Embarked"].mode()[0]
NewData["Embarked"] = NewData["Embarked"].fillna(Common)
NewData.isnull().sum()
#convert text to num
NewData["Sex"] = NewData["Sex"].map({'male' : 0 , 'female': 1})
NewData["Embarked"] = NewData["Embarked"].map({'S' :  0, 'C' : 1,'Q':2})
NewData.info()
Combinedcol = ['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']
X = NewData[Combinedcol]
Y = NewData["Survived"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
Model = LogisticRegression()
Model.fit(X,Y)
Prediction = Model.predict(X)
print("This is our Prediction , zero means died and one meand alive")
print()
print (Prediction)
# COUNT PREDICTIONS
survived = np.sum(Prediction == 1)
died = np.sum(Prediction == 0)
print()
print("SUMMARY OF PREDICTIONS:")
print("Total Survived Predicted (1) =", survived)
print("Total Died Predicted     (0) =", died)