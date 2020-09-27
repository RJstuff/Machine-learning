import pandas as pd
import numpy as np

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
dataset = pd.read_csv("diabetes.csv", header = None, names = col_names)

#print(dataset.head())

diabetes_data = dataset.iloc[1:]
print(diabetes_data.head())

feature_col = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

x = diabetes_data[feature_col]
print(x.head())

y = diabetes_data["label"]
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(max_iter=1000)

LogReg.fit(X_train, Y_train)

Y_pred = LogReg.predict(X_test)

#print(LogReg.score(X_test, Y_test))

from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(Y_test, Y_pred) # actual vs predicted

print(con_mat)

import seaborn as sn
import matplotlib.pyplot as plt
"""
fig, ax = plt.subplots()

sn.set(font_scale=1.5)

sn.heatmap(pd.DataFrame(con_mat), annot = True, fmt="g")
ax.xaxis.set_label_position("top")
plt.title("(Confusion Matrix)", y= 1.1)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()"""

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sn.heatmap(pd.DataFrame(con_mat), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

import pickle
import os

if not os.path.exists("Models"):
    os.makedirs("Models")

Model_path = "Models/logisitc_reg.sav"
pickle.dump(LogReg, open(Model_path, "wb"))

data = [[6, 0, 33.6, 50, 148, 72, 0.627]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']) 

#Predict On new Data
new_pred = LogReg.predict(df)
print(new_pred)
