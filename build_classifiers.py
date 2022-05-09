#!/usr/bin/env python3

#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__maintainer__ = 'Rob Hoelzle'
__script_name__ = 'build_classifiers.py'
__version__ = '0.0.0'
__profiling__ = 'True'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

###############################################################################
###############################################################################

## Function definitions

#plot confusion matrix
def plot_confusion_matrix(y, y_predict):
    """
    Generates a confusion matrix heatmap from test and predict Ys
    """
    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='mako', fmt='g')
    ax.set_xlabel('Predicted Adaptability')
    ax.set_ylabel('True Adaptability')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Low', 'Moderate', 'High'])
    ax.yaxis.set_ticklabels(['Low', 'Moderate', 'High'])

###############################################################################
###############################################################################

## Main script

#import data
df = pd.read_csv('data/students_adaptability_level_online_education.csv')

#integer level codes
gender = {'Boy': 0, 'Girl': 1}
age = {'1-5': 0, '6-10': 1, '11-15': 2, '16-20': 3, '21-25': 4, '26-30': 5}
education = {'School': 0, 'College': 1, 'University': 2}
institution = {'Non Government': 0, 'Government': 1}
yesno = {'No': 0, 'Yes': 1}
loadshed = {'Low': 0, 'High': 1}
finance = {'Poor': 0, 'Mid': 1, 'Rich': 2}
internet = {'Mobile Data': 0, 'Wifi': 1}
network = {'2G': 0, '3G': 1, '4G': 2}
duration = {'0': 0, '1-3': 1, '3-6': 2}
device = {'Mobile': 0, 'Tab': 1, 'Computer': 2}
adaptivity = {'Low': 0, 'Moderate': 1, 'High': 2}

#convert data to integer levels
coded_dict = {
    'Gender':[], #0=Boy,1=Girl
    'Age':[], #0=1-5, 1=6-10, 2=11-15, 3=16-20, 4=21-25, 5=26-30
    'Education Level':[], #0=School, 1=College, 2=University
    'Institution Type':[], #0=Non Government, 1=Government
    'IT Student':[], #0=No, 1=Yes
    'In Town':[], #0=No, 1=Yes
    'Load-shedding':[], #0=Low, 1=High
    'Financial Condition':[], #0=Poor, 1=Mid, 2=Rich
    'Internet Type':[], #0=Mobile Data, 1=Wifi
    'Network Type':[], #0=2G, 1=3G, 2=4G
    'Class Duration':[], #0=0, 1=1-3, 2=3-6
    'Self Lms':[], #0=No, 1=Yes
    'Device':[], #0=Mobile, 1=Tab, 2=Computer
    'Adaptivity Level':[] #0=Low, 1=Moderate, 2=High
}

for i in range(0,df.shape[0]):
    coded_dict['Gender'].append(gender[df['Gender'][i]])
    coded_dict['Age'].append(age[df['Age'][i]])
    coded_dict['Education Level'].append(education[df['Education Level'][i]])
    coded_dict['Institution Type'].append(institution[df['Institution Type'][i]])
    coded_dict['IT Student'].append(yesno[df['IT Student'][i]])
    coded_dict['In Town'].append(yesno[df['Location'][i]])
    coded_dict['Load-shedding'].append(loadshed[df['Load-shedding'][i]])
    coded_dict['Financial Condition'].append(finance[df['Financial Condition'][i]])
    coded_dict['Internet Type'].append(internet[df['Internet Type'][i]])
    coded_dict['Network Type'].append(network[df['Network Type'][i]])
    coded_dict['Class Duration'].append(duration[df['Class Duration'][i]])
    coded_dict['Self Lms'].append(yesno[df['Self Lms'][i]])
    coded_dict['Device'].append(device[df['Device'][i]])
    coded_dict['Adaptivity Level'].append(adaptivity[df['Adaptivity Level'][i]])
    
#convert data to numpy array for model building
coded_df = pd.DataFrame(coded_dict)
Y = coded_df['Adaptivity Level'].to_numpy()
X = coded_df.loc[:, coded_df.columns != 'Adaptivity Level'].to_numpy()

#standardize data, and split to train and test sets
xform = preprocessing.StandardScaler()
X_z = xform.fit(X).transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_z, Y, test_size=0.2, random_state=2)

#Train and test Logistic Regression
parameters = {'C': [0.01, 0.1, 1],
              'penalty': ['none', 'l2', 'l1', 'elasticnet'],
              'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
              'multi_class': ['multinomial']}
lr = LogisticRegression()
grid_search = GridSearchCV(lr, parameters, cv=50, verbose=0)
logreg_cv = grid_search.fit(X_train, Y_train)

print("Tuned hpyerparameters (best parameters):", logreg_cv.best_params_)
print("Train accuracy:", logreg_cv.best_score_)
print("Test accuracy:", logreg_cv.best_estimator_.score(X_test, Y_test))
yhat = logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

#Train and test K Nearest Neighbors
parameters = {'n_neighbors': list(range(1, 20)),
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, parameters, cv=50, verbose=0)
knn_cv = grid_search.fit(X_train, Y_train)

print("Tuned hpyerparameters (best parameters):", knn_cv.best_params_)
print("Train accuracy:", knn_cv.best_score_)
print("Test accuracy:", knn_cv.best_estimator_.score(X_test, Y_test))
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

#Train and test Decision Tree
parameters = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [2*n for n in range(1,10)],
              'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10]}
tree = DecisionTreeClassifier()
grid_search = GridSearchCV(tree, parameters, cv=50, verbose=0)
tree_cv = grid_search.fit(X_train, Y_train)

print("Tuned hpyerparameters (best parameters):", tree_cv.best_params_)
print("Train accuracy:", tree_cv.best_score_)
print("Test accuracy:", tree_cv.best_estimator_.score(X_test, Y_test))
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

#Train and test Support Vector Machine
parameters = {'C': [0.001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'degree': [2],
              'gamma': [0.001]}
svm = SVC()
grid_search = GridSearchCV(svm, parameters, cv=2, verbose=3)
svm_cv = grid_search.fit(X_train, Y_train)

print("Tuned hpyerparameters (best parameters):", svm_cv.best_params_)
print("Train accuracy:", svm_cv.best_score_)
print("Test accuracy:", svm_cv.best_estimator_.score(X_test, Y_test))
yhat = svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

#Compare all models
print("Logistic Regression: {}\nKNN: {}\nDecision Tree: {}\nSVM: {}".format(
    logreg_cv.best_estimator_.score(X_test, Y_test),
    knn_cv.best_estimator_.score(X_test, Y_test),
    tree_cv.best_estimator_.score(X_test, Y_test),
    svm_cv.best_estimator_.score(X_test, Y_test)))