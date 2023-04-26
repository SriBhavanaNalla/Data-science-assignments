import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('bank-full.csv', sep = ';')
df.head()
df.tail()
df.shape
df.info()

#check for missing values
df.isna().sum()

df.isna()

#to genrate heat map
import seaborn as sns
cols = df.columns
colors = ['#000099', '#ffff00']   #define colors, yellow for non missing, blue for missing
sns.heatmap(df[cols].isnull(),
               cmap= sns.color_palette(colors))

#check for duplicate values
df[df.duplicated()].shape
df[df.duplicated()]
df.dtypes

#to correct dtypes of columns
df['job'] = df['job'].astype('category')
df['marital'] = df['marital'].astype('category')
df['education'] = df['education'].astype('category')
df['default'] = df['default'].astype('category')
df['housing'] = df['housing'].astype('category')
df['loan'] = df['loan'].astype('category')
df['contact'] = df['contact'].astype('category')
df['month'] = df['month'].astype('category')
df['poutcome'] = df['poutcome'].astype('category')
df['y'] = df['y'].astype('category')
df.dtypes

#Get descreptive Stats of numerical columns
df.describe()
#Univariate Analysis
#for categorical columns
values_job = df.job.value_counts()
labels_job = df.job.unique().tolist()
plt.pie(values_job, labels=labels_job, radius = 2)

print(values_job)


values_marital = df.marital.value_counts()
labels_marital = df.marital.unique().tolist()
plt.pie(values_marital, labels=labels_marital, radius = 2)

print(values_marital)

values_education = df.education.value_counts()
labels_education = df.education.unique().tolist()
plt.pie(values_education, labels=labels_education, radius = 2)

print(values_education)


values_default = df.default.value_counts()
labels_default = df.default.unique().tolist()
plt.pie(values_default, labels=labels_default, radius = 2)

print(values_default)

values_contact = df.contact.value_counts()
labels_contact = df.contact.unique().tolist()
plt.pie(values_contact, labels=labels_contact, radius = 2)

print(values_contact)

values_housing = df.housing.value_counts()
labels_housing = df.housing.unique().tolist()
plt.pie(values_housing, labels=labels_housing, radius = 2)

print(values_housing)

values_loan = df.loan.value_counts()
labels_loan = df.loan.unique().tolist()
plt.pie(values_loan, labels=labels_loan, radius = 2)

print(values_loan)

values_y = df.y.value_counts()
labels_y = df.y.unique().tolist()
plt.pie(values_y, labels=labels_y, radius = 2)

print(values_y)

values_month = df.month.value_counts()
labels_month = df.month.unique().tolist()
plt.pie(values_month, labels=labels_month, radius = 2)

print(values_month)

values_poutcome = df.poutcome.value_counts()
labels_poutcome = df.poutcome.unique().tolist()
plt.pie(values_poutcome, labels=labels_poutcome, radius = 2)

print(values_poutcome)

#Bivariate Analysis
# first we will do for categorical columns with y
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.figure(figsize=(20,10))
sns.countplot(df.job, hue=df.y)

sns.countplot(df.marital, hue= df.y)
sns.countplot(df.education, hue = df.y)
sns.countplot(df.default, hue = df.y)
sns.countplot(df.housing, hue=df.y)
sns.countplot(df.loan, hue=df.y)
sns.countplot(df.contact, hue=df.y)
sns.countplot(df.month, hue=df.y)
sns.countplot(df.poutcome, hue = df.y)

# y Col to be transformed
df = pd.get_dummies(data=df, columns=['y'], drop_first= True)
df.head()

#now we will perform Bivariate analysis on numerical columns with y
plt.figure(figsize=(15,7))
sns.distplot(df[df['y_yes']==0]['age'])
sns.distplot(df[df['y_yes']==1]['age'])

print("amber - y_yes = 1, blue - y_yes = 0")

plt.figure(figsize=(15,7))
sns.distplot(df[df['y_yes']==0]['balance'])
sns.distplot(df[df['y_yes']==1]['balance'])

print("amber - y_yes = 1",'\n', "blue - y_yes = 0")

plt.figure(figsize=(15,7))
sns.countplot(df.day, hue = df.y_yes)

plt.figure(figsize=(15,7))
sns.distplot(df[df['y_yes']==0]['duration'])
sns.distplot(df[df['y_yes']==1]['duration'])

print("amber - y_yes = 1",'\n', "blue - y_yes = 0")

plt.figure(figsize=(15,7))
sns.distplot(df[df['y_yes']==0]['campaign'])
sns.distplot(df[df['y_yes']==1]['campaign'])

print("amber - y_yes = 1",'\n', "blue - y_yes = 0")

plt.figure(figsize=(15,7))
sns.distplot(df[df['y_yes']==0]['pdays'])
sns.distplot(df[df['y_yes']==1]['pdays'])

print("amber - y_yes = 1",'\n', "blue - y_yes = 0")

plt.figure(figsize=(15,7))
sns.distplot(df[df['y_yes']==0]['previous'])
sns.distplot(df[df['y_yes']==1]['previous'])

print("amber - y_yes = 1",'\n', "blue - y_yes = 0")

#checking for outliers
plt.boxplot(df.age)
plt.boxplot(df.duration)
plt.boxplot(df.campaign)

#Lets try diff transformation on age column
import numpy as np
plt.subplots(figsize = (9,6))
plt.suptitle('age')
plt.subplot(131)
plt.title('log')
plt.boxplot(np.log(df['age']))
plt.subplot(132)
plt.title('square root')
plt.boxplot(np.sqrt(df['age']))
plt.subplot(133)
plt.title('cuberoot')
plt.boxplot(np.cbrt(df['age']))

#Lets try diff transformation on age column
import numpy as np
plt.subplots(figsize = (9,6))
plt.suptitle('duration')
plt.subplot(131)
plt.title('log')
plt.boxplot(np.log(df['duration']))
plt.subplot(132)
plt.title('square root')
plt.boxplot(np.sqrt(df['duration']))
plt.subplot(133)
plt.title('cuberoot')
plt.boxplot(np.cbrt(df['duration']))

#Lets try diff transformation on age column
import numpy as np
plt.subplots(figsize = (9,6))
plt.suptitle('campaign')
plt.subplot(131)
plt.title('log')
plt.boxplot(np.log(df['campaign']))
plt.subplot(132)
plt.title('square root')
plt.boxplot(np.sqrt(df['campaign']))
plt.subplot(133)
plt.title('cuberoot')
plt.boxplot(np.cbrt(df['campaign']))
df.corr()

sns.pairplot(df)

df.dtypes

#to convert categorical columns data into dummy variables
df = pd.get_dummies(data=df, columns=['job'], drop_first= True)
df = pd.get_dummies(data=df, columns=['marital'], drop_first= True)
df = pd.get_dummies(data=df, columns=['education'], drop_first= True)
df = pd.get_dummies(data=df, columns=['default'], drop_first= True)
df = pd.get_dummies(data=df, columns=['housing'], drop_first= True)
df = pd.get_dummies(data=df, columns=['loan'], drop_first= True)
df = pd.get_dummies(data=df, columns=['contact'], drop_first= True)
df = pd.get_dummies(data=df, columns=['month'], drop_first= True)
df = pd.get_dummies(data=df, columns=['poutcome'], drop_first= True)
# To see all columns
pd.set_option("display.max.columns", None)
df
df.columns
df.dtypes

df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
       'y_yes', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_married', 'marital_single', 'education_secondary',
       'education_tertiary', 'education_unknown', 'default_yes', 'housing_yes',
       'loan_yes', 'contact_telephone', 'contact_unknown', 'month_aug',
       'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
       'poutcome_other', 'poutcome_success', 'poutcome_unknown']].astype('int64')
df.info()

df.dtypes
# Label Encoding Technique
order={'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}
df=df.replace(order)

#std scaler
from sklearn.preprocessing import StandardScaler

df_standard_scaled = df.copy()
column_name= ['balance','age','duration']
features = df_standard_scaled[column_name]
rest = [columns for columns in df.columns if columns not in column_name]

scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
df_standard_scaled = pd.DataFrame(features, columns = column_name)
df_standard_scaled[rest] = df[rest]
df_standard_scaled.head()

#To Fit a Model

# Dividing our data into input and output variables 
X = df_standard_scaled.iloc[:,1:42]
Y = df_standard_scaled.iloc[:,-1]
#Logistic regression and fit the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X,Y)

#LogisticRegression()
#Predict for X dataset
y_pred = classifier.predict(X)
y_pred_df= pd.DataFrame({'actual': Y,
                         'predicted_prob': classifier.predict(X)})
y_pred_df

## Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)

acc = (38961+961)/(38961+961+3488+1801)
print(acc)

#Classification report
from sklearn.metrics import classification_report
print(classification_report(Y,y_pred))


# ROC Curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, classifier.predict_proba (X)[:,1])

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')

auc
