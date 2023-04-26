from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
forest = pd.read_csv("forestfires.csv")
forest.info()

forest.T
forest.describe()
forest.isnull().sum()
forest.shape
numerical_feature = forest.describe(include=["int", "float"]).columns

print(list(numerical_feature))

import seaborn as sns
sns.set_style('whitegrid')
sns.pairplot(forest[numerical_feature])
plt.show()

categorical_features = forest.describe(include=["object"]).columns
print(list(categorical_features))
print(categorical_features)



for idx, column in enumerate(categorical_features):
    plt.figure(figsize=(15, 5))
    df = forest.copy()
    unique = df[column].value_counts(ascending=True);
 
    #plt.subplot(1, len(categorical_features), idx+1)    
    plt.title("Count of "+ column)
    sns.countplot(data=forest, x=column,palette = "dark")
    #plt.bar(unique.index, unique.values);
    plt.xticks(rotation = 90, size = 15)
    
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Number of "+ column, fontsize=12)
    plt.show()
    
    forest_1= forest.iloc[:,2:30]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(forest_1)
forest_norm = sc.transform(forest_1)
forest_norm

from sklearn.decomposition import PCA
pca = PCA(n_components = 28)
pca_values = pca.fit_transform(forest_norm)
pca_values

var = pca.explained_variance_ratio_
var
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
plt.figure(figsize=(12,4))
plt.plot(var1,color="green",marker = "P");


finalDf = pd.concat([pd.DataFrame(pca_values[:,0:24],columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7',
 'pc8','pc9','pc10','pc11','pc12','pc13','pc14',
 'pc15','pc16','pc17','pc18','pc19','pc20','pc21',
 'pc22','pc23','pc24']),
 forest[['size_category']]], axis = 1)
finalDf.size_category.replace(('large','small'),(1,0),inplace=True)
finalDf

import seaborn as sns
fig=plt.figure(figsize=(18,14))
sns.scatterplot(data=finalDf)

array = finalDf.values
X = array[:,0:24]
Y = array[:,24]
model = Sequential()
model.add(Dense(12, input_dim=24, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, validation_split=0.3, epochs=150, batch_size=10)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))