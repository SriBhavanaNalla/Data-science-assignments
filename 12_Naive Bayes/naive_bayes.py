import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.naive_bayes import GaussianNB

salary_train=pd.read_csv("SalaryData_Train.csv")
salary_test=pd.read_csv("SalaryData_Test.csv")
salary_train

string_col = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
labelencoder = LabelEncoder()

for i in string_col:
    salary_train[i]= labelencoder.fit_transform( salary_train[i])
    salary_test[i]=labelencoder.fit_transform(salary_test[i])
salary_train.head()

mapping = {' >50K': 1, ' <=50K': 2}
salary_train = salary_train.replace({'Salary': mapping})
salary_test = salary_test.replace({'Salary': mapping})
salary_train.head()

#Naive Bayes
x_train = salary_train.iloc[:,0:13]
y_train = salary_train.iloc[:,13]
x_test = salary_test.iloc[:,0:13]
y_test = salary_test.iloc[:,13]

#Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(x_train,y_train)

y_pred_mb=classifier_mb.predict(x_test)
cm=confusion_matrix(y_test,y_pred_mb)
cm

print ("Accuracy",np.mean(y_pred_mb==y_test.values.flatten()))


#Gaussian Naive Bayes
classifier_gb = GaussianNB()
classifier_gb.fit(x_train,y_train)

y_pred_gb=classifier_gb.predict(x_test)
cm=confusion_matrix(y_test,y_pred_gb)
cm

print ("Accuracy",np.mean(y_pred_gb==y_test.values.flatten()))

##GaussianNB Model has a better Accuracy