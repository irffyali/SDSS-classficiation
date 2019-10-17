# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:43:29 2019

@author: irffy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
sdss = pd.read_csv('C:/Users/irffy/OneDrive/Documents/side projetcs/SDSSdata.csv')
sns.countplot(x= "class", data = sdss)
sns.pairplot(sdss.loc[:,["u", "g", "r", "i", "z", "class"]], hue = "class") #exploratory analysis
plt.show()


def condition(x):
    if x == "STAR":
        return 1
    elif x == "GALAXY":
        return 2
    elif x == "QSO":
        return 3
sdss['classval'] = sdss['class'].apply(condition) # change classes to numerical values
y = sdss["classval"].values
cols = ['u','g','r','i','z'] #sleecting our predicitor variables 
x = sdss[cols].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30) #splitting train/test data
#comparing classifiers
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # Number of neighbors to consider.
knn.fit(x_train, y_train)
knnScore = knn.score(x_test, y_test)
print("Score of KNN Regression : {0}".format(knnScore))

scores = []
for each in range(1, 20):
    bestk = KNeighborsClassifier(n_neighbors = each) #locaring the best k value
    bestk.fit(x_train, y_train)
    scores.append(bestk.score(x_test, y_test))
    
plt.plot(range(1, 20), scores)
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.show();

from sklearn.metrics import classification_report,confusion_matrix




from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print(confusion_matrix(y_test,y_pred))
print("svm Accuracy:",metrics.accuracy_score(y_test, y_pred))



from sklearn import tree 

clf = tree.DecisionTreeClassifier()

dtree = clf.fit(x_train, y_train)
pred  = dtree.predict(x_test)

print("svm Accuracy:",metrics.accuracy_score(y_test, pred))