## Pytyhon Libraries 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

## Datasets
df_train=pd.read_csv('train.csv')

df_test=pd.read_csv('test.csv')

## Merge Datasets with Concat
df = pd.concat([df_train, df_test], ignore_index=True)

## Cleaning Dataset
df.drop(['User_ID'],axis=1,inplace=True)

df['Gender']=df['Gender'].map({'F':0,'M':1,np.nan:1})

df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7,np.nan:1})

df['City_Category']=df['City_Category'].map({'A':1,'B':2,'C':3,np.nan:1})

df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])

df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])

df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')

df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].fillna(df['Stay_In_Current_City_Years'].mode()[0])

df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)

## Visualiuzation
![image](https://github.com/user-attachments/assets/9b259589-8272-4f55-b287-8aa67d167886)
![image](https://github.com/user-attachments/assets/b8ffe59b-e6e0-4536-b74f-e086b140ab68)
![image](https://github.com/user-attachments/assets/7c4e496c-115b-42d0-8521-a4e29e1fd2b2)
![image](https://github.com/user-attachments/assets/ee8df478-019b-486b-ac9a-0d9e60e334d7)
![image](https://github.com/user-attachments/assets/7809218f-ec21-4bde-80cb-029f7db1c4c7)
![image](https://github.com/user-attachments/assets/1f261805-d83d-444a-85d4-5d691542d23e)

## Feature Scaling

X=df_train.drop('Purchase',axis=1)

Y=df_train['Purchase']


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)


((##**X:** Your feature data (all columns except 'Purchase').

**Y:** Your target variable ('Purchase').

**test_size=0.33:** This means 33% of the data will be used for testing, and the remaining 67% for training.

**random_state=42:** This ensures that the split is reproducible. You'll get the same split every time you run the code with the same random state.

The function returns four variables:

**X_train:** Training features

**X_test:** Testing features

**Y_train:** bold text Training target values

**Y_test:** Testing target values))


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)







