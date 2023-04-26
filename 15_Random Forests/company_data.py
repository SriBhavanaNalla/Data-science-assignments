# Bagging decision trees for classification
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

filename = 'Company_Data.csv'
names = ['Sales','CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']
dataframe = read_csv(filename)
dataframe

dataframe['Urban'].value_counts()
dataframe['Urban'] = dataframe['Urban'].map({'Yes': 1, 'No': 0})
dataframe['Urban'].value_counts()
dataframe['US'].value_counts()
dataframe['US'] = dataframe['US'].map({'Yes': 1, 'No': 0})
dataframe['US'].value_counts()

dataframe['ShelveLoc'].value_counts()
dataframe['ShelveLoc'] = dataframe['ShelveLoc'].map({'Bad': 0,'Medium': 1, 'Good': 2})
dataframe['ShelveLoc'].value_counts()

dataframe
array = dataframe.values
X = array[:,0:10]
Y = array[:,10]
seed = 7
kfold = KFold(n_splits = 10, random_state = seed, shuffle = True)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator = cart, n_estimators = num_trees, random_state = seed)
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())

results

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

kfold = KFold(n_splits = 10, random_state = 7, shuffle = True)
cart = DecisionTreeClassifier()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators = num_trees, max_features = max_features)
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())

# Stacking Ensemble for classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = KFold(n_splits = 10, random_state = 7, shuffle = True)

# create the sub models
estimators = []
model1 = LogisticRegression(max_iter = 500)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv = kfold)
print(results.mean())

estimators
model1
model2
model3
ensemble
