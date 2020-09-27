#ALWAYS COPY YOUR DATASET TO THE SCRIPTS OF YOUR MACHINE LEARNING CODES
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Salary_Data.csv")
print(dataset)

#create a matrix with independant variables
x = dataset.iloc[:,:-1].values
print(x)

#create a dependant vector
y = dataset.iloc[:,1].values
print(y)

#spliting dataset into train and test sets
from sklearn.model_selection import train_test_split #cross_validation did not work hence we used model_selection
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

#features scaling ---->standardization
""" 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) 
print(x_train)

"""
#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print(regressor.fit(x_train,y_train)) #regressor(machine) is created and fitted(learnt) into the training set

#Predicting the test set results
y_pred = regressor.predict(x_test) #matrix that contains predicted salaries of employees with help of x_test

#visualising the training set results
plt.scatter(x_test,y_test,color = 'green')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
