import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pydot

##Importing Company_Data data
data = pd.read_csv('Company_Data.csv')
data.head(10)

data.dtypes
data['High']= data.Sales.map(lambda x: 1 if x>7.5 else 0)

###converting categorical data
data['ShelveLoc']=data['ShelveLoc'].astype('category')
data['Urban']=data['Urban'].astype('category')
data['US']=data['US'].astype('category')
data.dtypes

##converting category to numeric data
data['ShelveLoc']=data['ShelveLoc'].cat.codes
data['Urban']=data['Urban'].cat.codes
data['US']=data['US'].cat.codes
data.dtypes
data.head()

##Model building
label_encoder = preprocessing.LabelEncoder()
X = data.iloc[:,1:11]
Y = data.iloc[:,11]
print(X)
print(Y)

data['High'].unique()
data.High.value_counts()

##Splting data into trainig and testing data set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 40)
##Build decision tree classifire using entropy criteria
model = DecisionTreeClassifier(criterion ='entropy',max_depth=3)
model.fit(X_train,Y_train)

##plot the decission tree
tree.plot_tree(model);

from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion = 'gini', max_depth = 3)
model_gini.fit(X_train,Y_train)

## predict the accuracy 
pred = model_gini.predict(X_test)
np.mean(pred == Y_test)

# Decision tree as regration 
# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
array = data.values
x = array[:,1:11]
y = array[:,11]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.33, random_state =50)
model = DecisionTreeRegressor()
model.fit(x_train,y_train)

pred = model.predict(x_test)
pred

# find the accuracy
model.score(x_test,y_test)

################################################################


##FRAUD CHECK
file = pd.read_csv('Fraud_check.csv')
file.head()
file1 = file.rename({'Marital.Status':'M','Taxable.Income':'income','City.Population':'population','Work.Experience':'work'},axis = 1)
file1.info()
file1.head()

#Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
file1= pd.get_dummies(file1,columns=['Undergrad','M','Urban'],drop_first = True)
file1.head()

#Creating new cols TaxInc for Risky and Good
file1["Tax"] = pd.cut(file1["income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
#Lets assume: taxable_income <= 30000 as “Risky=0” and others are “Good=1”
#After creation of new col. TaxInc also made its dummies var concating right side of df
file1 = pd.get_dummies(file1,columns = ["Tax"],drop_first=True)
file1.tail(10)

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(file1.iloc[:,1:])
df_norm.tail(10)

x = df_norm.drop(['Tax_Good'], axis=1)
y = df_norm['Tax_Good']
from sklearn.model_selection import train_test_split
# Splitting data into train & test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model_gini1 = DecisionTreeRegressor()
model_gini1.fit(x_train,y_train)

# prddiction and computing the accuracy
pred = model_gini1.predict(x_test)
np.mean(pred== y_test)

# Random forest classification 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

num_trees = 100
max_featues = 3

kfold =KFold(n_splits =10, random_state = 7,shuffle=True)
model = RandomForestClassifier(n_estimators = num_trees, max_features = max_featues)
results = cross_val_score(model,x,y,cv =kfold)
print(results.mean())