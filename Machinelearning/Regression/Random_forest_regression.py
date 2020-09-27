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

#fitting a random forest regression model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x,y)

'''#predicting a new result with random forest regression
y_pred = regressor.predict(6.5)
print(y_pred)
'''
#visualising the random forest regression results (for higher resolution)
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='blue')
plt.plot(x_grid,regressor.predict(x_grid),color='red')
plt.title('truth or bluff')
plt.xlabel('position levels')
plt.ylabel('Salaries')
plt.show()

