import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
data = pd.read_csv("student-mat.csv" ,sep=";")

data = data[["G1","G2","G3","studytime","failures","absences"]]

x = np.array(data.drop(["G3"],1))

y = np.array(data["G3"])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

best = 0
'''
for i in range(30):
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    accuracy = linear.score(x_test,y_test)
    print("Model's Accuracy:" , accuracy*100,"%")

    if accuracy > best:
        best = accuracy
        with open("StudentModel.pickle", "wb") as f:
            pickle.dump(linear, f) ''' #Do not need to train model anymore

pickle_in = open("StudentModel.pickle", "rb")
linear = pickle.load(pickle_in)

print("cofficients : ", linear.coef_)
print("interception: ", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):

    print(predictions[x], x_test[x], y_test[x])

style.use("ggplot")

pyplot.scatter(data["G1"],data["G3"])
pyplot.xlabel("G1")
pyplot.ylabel("Final Grade : G3")
pyplot.show() 
