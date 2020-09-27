#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")
print(dataset)

#create a matrix with independant variables
x = dataset.iloc[:,1:2].values
print(x)

#create a dependant vector
y = dataset.iloc[:,2].values
print(y)

#spliting dataset into train and test sets
"""
from sklearn.model_selection import train_test_split #cross_validation did not work hence we used model_selection
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
"""
"""
#features scaling ---->standardization

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) 
print(x_train)
"""
"""
#fitting a linear regression model to dataset
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(x,y)

#visualising linear regression model result
plt.scatter(x,y,color='blue')
plt.plot(x,lin_reg1.predict(x),color='red')
plt.title('')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
"""

#fitting a polynomial regression to dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
print(x_poly)

#visualising the polynomial regression result
plt.scatter(x,y,color='blue')
plt.plot(x,lin_reg.predict(poly_reg.fit_transform(x)),color='red')
plt.title('truth or bluff')
plt.xlabel('position levels')
plt.ylabel('salary')
plt.show()

