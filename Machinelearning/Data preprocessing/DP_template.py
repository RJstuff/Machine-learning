#ALWAYS COPY YOUR DATASET TO THE SCRIPTS OF YOUR MACHINE LEARNING CODES
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

#spliting dataset into train and test sets
from sklearn.model_selection import train_test_split #cross_validation did not work hence we used model_selection
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
print(y_test)
print(x_test)

#features scaling ---->standardization
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) 
print(x_train)
"""

