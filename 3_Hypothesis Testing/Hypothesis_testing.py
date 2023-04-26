#1)
import pandas as pd

import numpy as np
from scipy import stats
data=pd.read_csv("Cutlets.csv")
data.head()
p_value=stats.ttest_ind(data.iloc[:,0],data.iloc[:,1])
p_value

## accept null hypothesis as p value is more than alpha 0.05

#==========================================================

#2)
data1=pd.read_csv("LabTAT.csv")
data1.head()
p_value=stats.f_oneway(data1.iloc[:,0],data1.iloc[:,1],data1.iloc[:,2],data1.iloc[:,3])
p_value
#F_onewayResult(statistic=118.70421654401437, pvalue=2.1156708949992414e-57)

## value of p value i.e. 2.11*10^-57  which is less than alpha 0.05 Reject Null Hypothesis

#==============================================================

#3)
data2=pd.read_csv("BuyerRatio.csv")
data2.head()
X=pd.Series(data2.iloc[0,1:]).values
X
Y=pd.Series(data2.iloc[1,1:]).values
Y
z=[X,Y]
z

from scipy.stats import chi2_contingency
x=chi2_contingency(z)
x[1]
# 0.6603094907091882
#p value is greater than Alpha value 0.05 therefore we will Accept Null Hypothesis

#=================================================

#4)
data3=pd.read_csv("Costomer+OrderForm.csv")
data3.head()
data3.Phillippines.value_counts()
data3.Indonesia.value_counts()
data3.Malta.value_counts()
data3.India.value_counts()
array = [[271,267,269,280],[29,33,31,20]]
array
x=chi2_contingency(array)
x[1]
#0.2771020991233135
# p value is greater than alpha value therefore we will accept null hypothesis


#====================================================================================








