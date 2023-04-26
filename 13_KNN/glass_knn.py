# KNN Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
#%matplotlib inline
from sklearn.model_selection import train_test_split, cross_val_score
# Grid Search for Algorithm Tuning
import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
#read data
df=pd.read_csv("glass.csv")
df

X = np.array(df.iloc[:, 3:5])
Y = np.array(df['Type'])

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)
n_neighbors = numpy.array(range(1,30))
param_grid = dict(n_neighbors=n_neighbors)
model = KNeighborsClassifier()          
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)


k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

model =  KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)

pred = model.predict(X_test)
score = accuracy_score(pred,y_test)
score