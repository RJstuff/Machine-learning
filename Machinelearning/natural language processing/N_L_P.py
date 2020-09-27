#Natural language processing

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing dataset which is in tsv format
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3) #to ignore the double quotes ,we'll encode value of quote =3
#print(dataset)

#Cleaning the texts - 'the','a','and' etc. because these are not the words that will help our model to predict if restaurent is good or not

"""import re  # for cleaning text
import nltk
from nltk.corpus import stopwords

review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0]) # remove all things except alphabets
review = review.lower()
review = review.split() #convert to list
review = [word for word in review if not word in set(stopwords.words('english'))] # removing 'this' and set is used to  handle large paraagraphs 

#Stemming --> keep the word 'love' instead of 'loving','loved'
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

review = ' '.join(review)

""" # for one review
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] #empty list
for i in range(0,1000): #cleaning for all
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) # remove all things except alphabets
    review = review.lower()
    review = review.split() #convert to list
    review = [word for word in review if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)
    corpus.append(review)
#print(review)

#creating bag of words model using a process tokenizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray() #matrix of features
y = dataset.iloc[:,1].values #dependant variable

#Spliting it into test and training set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)

#fitting naive bayes to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

#predicting the test set result
y_pred = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)

print(cm)

import seaborn as sn


df_con = pd.DataFrame(cm, range(2), range(2))

sn.set(font_scale = 1.5)

sn.heatmap(df_con, annot = True, annot_kws = {"size":1.5})

plt.show()
    



    
