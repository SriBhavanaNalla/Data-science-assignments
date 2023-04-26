#1)
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
# import dataset

df=pd.read_csv("delivery_time.csv")
df
df.info()
sns.distplot(df['Delivery Time'])
sns.distplot(df['Sorting Time'])

# Renaming Columns
df=df.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
df

#correlation analysis
df.corr()
sns.regplot(x=df['sorting_time'],y=df['delivery_time'])

####Model Building
model=smf.ols("delivery_time~sorting_time",data=df).fit()

###Model Testing
# Finding Coefficient parameters
model.params

# Finding tvalues and pvalues
model.tvalues , model.pvalues

# Finding Rsquared Values
model.rsquared , model.rsquared_adj

##Model Predictions

# Manual prediction for say sorting time 5
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time

# Automatic Prediction for say sorting time 5, 8
new_data=pd.Series([5,8])
new_data

data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred

model.predict(data_pred)
#============================================

#2)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
salary=pd.read_csv('Salary_Data.csv')
salary

salary.head()
salary.shape
salary.dtypes
salary.describe()
salary.corr()
sns.regplot(x=salary.YearsExperience, y=salary.Salary)

import statsmodels.formula.api as smf
model = smf.ols("Salary~YearsExperience",data=salary).fit()
model.summary()

model.params
model1 = smf.ols("Salary~np.log(YearsExperience)",data=salary).fit()
model1.summary()

model1.params
model2=smf.ols("Salary~np.exp(YearsExperience)",data=salary).fit()
model2.summary()

model2.params
pred=model.predict(salary)
plt.scatter(x=salary.YearsExperience, y=salary.Salary, color='blue')
plt.plot(salary.YearsExperience, pred,color='black')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")


pred=model1.predict(salary)
plt.scatter(x=salary.YearsExperience, y=salary.Salary, color='blue')
plt.plot(salary.YearsExperience, pred,color='black')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")

pred=model2.predict(salary)
plt.scatter(x=salary.YearsExperience, y=salary.Salary, color='blue')
plt.plot(salary.YearsExperience, pred,color='black')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")

####################################################

X = df[['sorting_time']] # Only Independent variables
Y = df["delivery_time"]  # Only Dependent variable

#step 4: model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

#step 5:predicting x values
Y_pred=LR.predict(X)

#step6:mean square error
from sklearn.metrics import mean_squared_error,r2_score
MSE = mean_squared_error(Y,Y_pred)
MSE

import numpy as np
RMSE = np.sqrt(MSE)
print("Root mean square error:" , RMSE.round(2))










