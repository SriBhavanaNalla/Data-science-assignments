import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot

#Importing the Data
toyota= pd.read_csv("ToyotaCorolla.csv",encoding='latin1')
toyota.head(6)

toyota_data = toyota.loc[:,["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
toyota_data

#Data Understanding
toyota_data.shape

toyota_data.isna().sum()

toyota_data[toyota_data.duplicated()]
toyota_data=toyota_data.drop_duplicates().reset_index(drop=True)
toyota_data
toyota_data.describe()

#3.[i]) Check For Assumption (Linearity)
sns.pairplot(toyota_data)

#3.[iii]) Checking Multicollinearity
corr =toyota_data.corr()
plt.figure(figsize=(12,10))  # it shows that there is  no Multicollinearity in the variables between eachother (values are near to zero)
sns.heatmap(corr,annot=True)
plt.show()

rsq_Age_08_04= smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_data).fit().rsquared
vif_Age_08_04 = 1/(1-rsq_Age_08_04)

rsq_KM= smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_data).fit().rsquared
vif_KM = 1/(1-rsq_KM)

rsq_HP= smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_data).fit().rsquared
vif_HP = 1/(1-rsq_HP)

rsq_cc= smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=toyota_data).fit().rsquared
vif_cc = 1/(1-rsq_cc)

rsq_Doors= smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toyota_data).fit().rsquared
vif_Doors = 1/(1-rsq_Doors)

rsq_Gears= smf.ols('Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight',data=toyota_data).fit().rsquared
vif_Gears = 1/(1-rsq_Gears)

rsq_Quarterly_Tax= smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight',data=toyota_data).fit().rsquared
vif_Quarterly_Tax = 1/(1-rsq_Quarterly_Tax)

rsq_Weight= smf.ols('Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=toyota_data).fit().rsquared
vif_Weight= 1/(1-rsq_Weight)

dv={'Variables':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'],
    'Vif':[vif_Age_08_04,vif_KM,vif_HP,vif_cc,vif_Doors,vif_Gears,vif_Quarterly_Tax,vif_Weight]}
Vif=pd.DataFrame(dv)
Vif


#3.[iv]Check Autoregression
#No AutoRegrrssion (No Time influence)

#3.[v]Check Zero residual Mean
sns.lmplot(x='Price',y='Age_08_04',data=toyota_data)
plt.show()

sns.lmplot(x='Price',y='KM',data=toyota_data)   
plt.show()

sns.lmplot(x='Price',y='HP',data=toyota_data)
plt.show()

sns.lmplot(x='Price',y='cc',data=toyota_data)
plt.show()

sns.lmplot(x='Price',y='Doors',data=toyota_data)
plt.show()

sns.lmplot(x='Price',y='Gears',data=toyota_data)
plt.show()

sns.lmplot(x='Price',y='Quarterly_Tax',data=toyota_data)
plt.show()

sns.lmplot(x='Price',y='Weight',data=toyota_data)
plt.show()

#Data Preparation
toyota_data.head(6)
X = toyota_data.drop(['Price'], axis=1)
X
y = toyota_data.iloc[:,:1]
y

#Model Builing and Model Training
#from sklearn.linear_model import LinearRegression
# Training with Sklearn
mult_linear_model = LinearRegression()

# training with Stats Models
mult_linear_model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_data).fit()
#mult_linear_model.fit(X,y)
y_pred = mult_linear_model.predict(X)
y_pred = pd.DataFrame(y_pred)
y_pred


#Model Evaluation
mult_linear_model.summary()

m_cc =smf.ols('Price~cc',data=toyota_data).fit()
m_cc.summary()

m_Doors = smf.ols('Price~Doors',data=toyota_data).fit()
m_Doors.summary()

m_cc_Doors = smf.ols('Price~Doors+cc',data=toyota_data).fit()
m_cc_Doors.summary()



from sklearn.metrics import mean_squared_error
mean_squared_error(y_true=y, y_pred=y_pred)

def standard_values(vals) :
    return (vals-vals.mean())/vals.std()

sns.scatterplot(standard_values(mult_linear_model.fittedvalues),standard_values(mult_linear_model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()

#Residuals Vs Regressors Plot
fig=plt.figure(figsize=(14,10))
sm.graphics.plot_regress_exog(mult_linear_model,'Age_08_04',fig=fig)
plt.show()

fig=plt.figure(figsize=(14,10))
sm.graphics.plot_regress_exog(mult_linear_model,'KM',fig=fig)
plt.show()

fig=plt.figure(figsize=(14,10))
sm.graphics.plot_regress_exog(mult_linear_model,'HP',fig=fig)
plt.show()

fig=plt.figure(figsize=(14,10))
sm.graphics.plot_regress_exog(mult_linear_model,'cc',fig=fig)
plt.show()

fig=plt.figure(figsize=(14,10))
sm.graphics.plot_regress_exog(mult_linear_model,'Doors',fig=fig)
plt.show()

fig=plt.figure(figsize=(14,10))
sm.graphics.plot_regress_exog(mult_linear_model,'Gears',fig=fig)
plt.show()

fig=plt.figure(figsize=(14,10))
sm.graphics.plot_regress_exog(mult_linear_model,'Quarterly_Tax',fig=fig)
plt.show()


fig=plt.figure(figsize=(14,10))
sm.graphics.plot_regress_exog(mult_linear_model,'Weight',fig=fig)
plt.show()

#Finding and removing Outlies to increase the model accuracy
(c,_) =mult_linear_model.get_influence().cooks_distance
c

fig=plt.figure(figsize=(20,7))
sns.pointplot(np.arange(len(toyota_data)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()

np.argmax(c), np.max(c)


#influencer plot
fig=plt.figure(figsize=(14,10))
sm.graphics.influence_plot(mult_linear_model)
plt.show()

#Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features(columns) & n = no. of observation
k=toyota_data.shape[1]
n=toyota_data.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff

#From the above plot, it is clear that points beyond leverage cutoff value=0.020905 are the outliers
#Model Improvement

toyota_data_d1=toyota_data.drop(labels=80,axis=0).reset_index(drop=True)
toyota_data_d1

X = toyota_data_d1.drop(['Price'], axis= 1)
y = toyota_data_d1.iloc[:,:1]
mult_linear_model_m1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_data_d1).fit()
y_pred = mult_linear_model_m1.predict(X)
y_pred = pd.DataFrame(y_pred)
y_pred


mult_linear_model_m1.summary()


mean_squared_error(y,y_pred)


#In improved Model Value of AIC and BIC is reduced.Mean squared error is also reduced. Adj. R sqaured value is now 0.867 means our model is 86.7 % accurate which more than first model.
mult_linear_model.rsquared_adj

#Adj. R squared Value is 86.17 % so all the feature have 86.17% contribution on predicted values

from sklearn.metrics import mean_squared_error
mean_squared_error(y_true=y, y_pred=y_pred)

#Model Improvement with Linear Regression
#With sqrt Transformation
toyota_data['sqrt_Age_08_04'] = np.sqrt(toyota_data['Age_08_04']) 
toyota_data['sqrt_KM'] = np.sqrt(toyota_data['KM'])
toyota_data['sqrt_HP'] = np.sqrt(toyota_data['HP'])
toyota_data['sqrt_cc'] = np.sqrt(toyota_data['cc'])
toyota_data['sqrt_Doors'] = np.sqrt(toyota_data['Doors'])
toyota_data['sqrt_Gears'] = np.sqrt(toyota_data['Gears'])
toyota_data['sqrt_Quarterly_Tax'] = np.sqrt(toyota_data['Quarterly_Tax'])
toyota_data['sqrt_Weight'] = np.sqrt(toyota_data['Weight'])
toyota_data_d2=toyota_data.drop (['Age_08_04','KM',	'HP',	'cc'	,'Doors',	'Gears'	,'Quarterly_Tax',	'Weight'],axis=1).drop(labels=80,axis=0).reset_index(drop=True)
toyota_data_d2

sns.pairplot(toyota_data_d2)
plt.show()


toyota_data_d2.corr()

X = toyota_data_d2.loc[:,["sqrt_Age_08_04","sqrt_KM","sqrt_HP",'sqrt_cc','sqrt_Doors',	'sqrt_Gears',	'sqrt_Quarterly_Tax',"sqrt_Weight"]]
mult_linear_model_m2 =smf.ols('Price~sqrt_Age_08_04+sqrt_KM+sqrt_HP+sqrt_cc+sqrt_Doors+sqrt_Gears+sqrt_Quarterly_Tax+sqrt_Weight',data=toyota_data_d2).fit()
y_pred = mult_linear_model_m2.predict(X)
pd.DataFrame(y_pred)

mult_linear_model_m2.summary()

#Model Evaluation
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true=y, y_pred=y_pred)

#Model Improvement
(c,_) =mult_linear_model_m2.get_influence().cooks_distance
c

fig=plt.figure(figsize=(20,7))
sns.pointplot(np.arange(len(toyota_data_d2)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()

toyota_data_d3= toyota_data_d2.drop(toyota_data_d2.index[c>0.5],axis=0).reset_index(drop=True)
toyota_data_d3

X = toyota_data_d3.loc[:,["sqrt_Age_08_04","sqrt_KM","sqrt_HP",'sqrt_cc','sqrt_Doors',	'sqrt_Gears',	'sqrt_Quarterly_Tax',"sqrt_Weight"]]
mult_linear_model_m3 =smf.ols('Price~sqrt_Age_08_04+sqrt_KM+sqrt_HP+sqrt_cc+sqrt_Doors+sqrt_Gears+sqrt_Quarterly_Tax+sqrt_Weight',data=toyota_data_d3).fit()
y_pred = mult_linear_model_m3.predict(X)
pd.DataFrame(y_pred)
mult_linear_model_m3.summary()