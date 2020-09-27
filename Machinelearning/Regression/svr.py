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

#features scaling --->standardization
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#fitting a svr model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

#predicting a new result with polynomial regression


#visualising the regression results

plt.scatter(x,y,color='blue')
plt.plot(x,regressor.predict(x),color='red')
plt.title('truth or bluff')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

"""
#visualising the svr results (for higher resolution)
x_grid = np.arange(min(x),max(x),0.1)
x_grid = np.reshape((len(x_grid),1))
plt.scatter(x,y,color='blue')
plt.plot(x_grid,regressor.predict(x_grid),color='red')
plt.xlabel()
plt.ylabel()
plt.show()
"""
