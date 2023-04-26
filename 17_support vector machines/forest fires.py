import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Feature engineering Libraries
from sklearn.preprocessing  import MinMaxScaler
from sklearn.model_selection import train_test_split

# Model Building Libraries
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import  accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

#import WARNING
import warnings
forestfire = pd.read_csv('forestfires.csv')
df = forestfire.copy()
df.head()
df.info()

df.shape
df.isnull().sum()
df.describe()

df.duplicated().sum()

df = df.drop_duplicates()
df.shape

df_num_data = df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']]
df_num_data.corr()

size_dummy = pd.get_dummies(df['size_category'], drop_first=True)
size_dummy

df1 = pd.concat([df,size_dummy],axis=1)
df1.head()
sns.heatmap(df_num_data.corr(), annot=True)
df1.columns
sns.countplot(df1['month'],hue=df1['size_category'])
sns.countplot(df1['day'],hue=df1['size_category'])
sns.countplot(df1['size_category'])
sns.displot(df['FFMC'],kde=True, bins=35)
sns.displot(df['DMC'],kde=True)
sns.displot(df['DC'],kde=True)

sns.displot(df['ISI'],kde=True, bins=25)
sns.displot(df['area'],kde=True, bins=15)
sns.displot(df['rain'],kde=True)
sns.displot(df['RH'],kde=True)
sns.displot(df['temp'],kde=True)
sns.displot(df['wind'],kde=True)
df1.columns

pp_df1 = pd.concat([df_num_data, df1], axis=1)
df1.head()
pp_df1.head()
sns.boxplot(df1['small'],df1['wind'])
sns.boxplot(df1['small'],df1['RH'])
sns.boxplot(df1['small'],df1['FFMC'])
sns.boxplot(df1['small'],df1['DMC'])
sns.boxplot(df1['small'],df1['DC'])
sns.boxplot(df1['small'],df1['ISI'])
sns.boxplot(df1['small'],df1['temp'])
sns.boxplot(df1['small'],df1['area'])
sns.boxplot(df1['small'],df1['rain'])




df1.head(2)
X = df1.drop(['month','day','size_category', 'small'],axis=1)
y= df1['small']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
scalar = MinMaxScaler()
scalar.fit(X_train)

scaled_X_train = scalar.transform(X_train)
scaled_X_test = scalar.transform(X_test)
#Converting Numpy arrays into DataFrame
scaled_X_train = pd.DataFrame(scaled_X_train,columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaled_X_test,columns=X_test.columns)
scaled_X_train.shape, scaled_X_test.shape, y_train.shape, y_test.shape

##GridSearch CV|
rdf_clf= SVC()
rdf_param_grid= {'kernel': ['rbf', 'poly', 'linear'], 'gamma': [50, 10, 6, 5, 4, 0.5], 'degree': [2,3,4], 'C': [15,14,13,12,11,10,1,0.1,0.01,0.001]}
rdf_gsv= GridSearchCV(rdf_clf, param_grid=rdf_param_grid, cv=10)
rdf_gsv.fit(scaled_X_train, y_train)

#printing the best scores of GSV
rdf_gsv.best_params_, rdf_gsv.best_score_

#Using Poly Kernel
# modeling with best parameters
poly_clf= SVC(kernel='poly', C= 10, degree=2, gamma=10)
poly_clf.fit(scaled_X_train, y_train)

y_pred= poly_clf.predict(scaled_X_test)
acc= accuracy_score(y_test, y_pred)
print('Accuracy :', acc)
confusion_matrix(y_test, y_pred)