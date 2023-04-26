import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

#import WARNING
import warnings
warnings.filterwarnings('ignore')
df1 = pd.read_csv('SalaryData_Train.csv')
df2 = pd.read_csv('SalaryData_Test.csv')
df1.head()
df2.head()
df1.info()
df2.info()
df1.shape
df2.shape

df1.isnull().sum()
df2.isnull().sum()
df1.duplicated().sum()
df2.duplicated().sum()

df2 = df2.drop_duplicates()
df2.duplicated().sum()
df1.describe()
df1['Salary1']= pd.get_dummies(df1['Salary'], drop_first=True)
df1.head()
df1.corr()
df1.corr()['Salary1']

##EDA
df1.columns
features = ['age', 'workclass', 'education', 'educationno', 'maritalstatus',
       'occupation', 'relationship', 'race', 'sex', 'capitalgain',
       'capitalloss', 'hoursperweek', 'native', 'Salary']
for feature in features:
    unique_features = df1[feature].unique()
    num_features = df1[feature].nunique()
    
    if num_features <= 10:
        print('{} has {} values as {}'.format(feature, num_features, unique_features))
    else:
        print('{} has {} values as {}......'.format(feature, num_features, unique_features[0:10]))

df1['education'].unique()
df1['educationno'].unique()

##Droping Education Column from Data
df1 = df1.drop('education', axis=1)
df2 = df2.drop('education', axis=1)
df1.columns
sns.countplot(df1['maritalstatus'],hue=df1['Salary'])
sns.countplot(df1['race'],hue=df1['Salary'])
sns.countplot(df1['sex'],hue=df1['Salary'])
sns.countplot(df1['workclass'],hue=df1['Salary'])
sns.countplot(df1['occupation'],hue=df1['Salary'])

sns.distplot(df1['age'])
sns.boxplot(df1['age'])
sns.distplot(df1['capitalgain'])
sns.boxplot(df1['capitalgain'])
sns.distplot(df1['capitalloss'])

sns.boxplot(df1['capitalloss'])
sns.distplot(df1['hoursperweek'])
sns.boxplot(df1['hoursperweek'])
sns.countplot(df1['Salary'])
df1['Salary'].value_counts().plot(kind= 'pie')
sns.pairplot(df1)

#Split Data
X = df1.drop(['Salary','Salary1'], axis=1)
y = df1['Salary']
X.head()
y.head()
y.value_counts()

#Pipeline for Preporcessing and Calassification
X.columns

#Numerical Transformation
num_col = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
Numerical_Trans = MinMaxScaler()
cat_col = ['workclass','educationno','maritalstatus', 'occupation','relationship', 'race', 'sex', 'native']
Catigorical_Trans = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(transformers=[('num',Numerical_Trans, num_col),('cat',Catigorical_Trans,cat_col)])
#Pipline
pipe = Pipeline(steps=[('preprocessor', preprocessor)])
x = pipe.fit_transform(X)
x = x.todense()
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
clf = SVC()
#fit
clf.fit(x_train,y_train)
#model score in 3 digit after decimal
print('Model Scroe: %.3f' % clf.score(x_test,y_test))


##Grid serarch CV
poly_param = {'kernel': ['poly'], 'gamma': [1, 0.5], 'degree': [2], 'C': [1]}
poly_grid= GridSearchCV(clf, param_grid=poly_param, cv=3)
poly_grid.fit(x_train,y_train)
poly_grid.best_params_, poly_grid.best_score_
