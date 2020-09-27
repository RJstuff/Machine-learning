#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("d.csv")
print(dataset)

#create a matrix with independant variables
x = dataset.iloc[:,:-1].values
print(x)

#create a dependant vector
y = dataset.iloc[:,3].values
print(y)

#taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)

#encoding categorial data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l_x = LabelEncoder() #creating a obj
x[:,0] = l_x.fit_transform(x[:,0]) #to apply obj on 1st column of matix x 

ohe = OneHotEncoder(categorical_features = [0])
x = ohe.fit_transform(x).toarray() 
print(x)
l_y = LabelEncoder()
y = l_y.fit_transform(y)
print(y)

#spliting dataset into train and test sets
from sklearn.model_selection import train_test_split #cross_validation did not work hence we used model_selection
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
print(y_test)

#features scaling ---->standardization
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) 
print(x_train)























