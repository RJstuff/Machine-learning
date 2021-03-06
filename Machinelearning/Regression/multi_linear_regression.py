#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#encoding categorial data
#encoding the independant variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#splitting datasets into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fitting multilinear regression into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(x_train)

#predict the test set results
y_pred = regressor.predict(x_test)
print(y_pred)

plt.plot(y_pred,y_test,color = 'red')
plt.scatter(x_train,regressor.predict(x_train),color = 'blue')
plt.title('profit vs multi independant variabels')
plt.xlabel('independant variabels')
plt.ylabel('profit')
plt.show()
