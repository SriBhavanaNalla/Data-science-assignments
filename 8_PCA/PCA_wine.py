# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
# Import Dataset
wine=pd.read_csv('wine.csv')
wine

wine['Type'].value_counts()
wine2=wine.iloc[:,1:]
wine2
wine2.shape
wine2.info()

# Converting data to numpy array
wine_ary=wine2.values
wine_ary

# Normalizing the numerical data 
wine_norm=scale(wine_ary)
wine_norm

##-----------PCA Implementation
# Applying PCA Fit Transform to dataset
pca=PCA(n_components=13)

wine_pca=pca.fit_transform(wine_norm)
wine_pca

# PCA Components matrix or covariance Matrix
pca.components_

# The amount of variance that each PCA has
var=pca.explained_variance_ratio_
var

# Cummulative variance of each PCA
var1=np.cumsum(np.round(var,4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color='magenta')

# Final Dataframe
final_df=pd.concat([wine['Type'],pd.DataFrame(wine_pca[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df

# Visualization of PCAs
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df)

######------------1. Hierarchical Clustering
# Import Libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
# As we already have normalized data, create Dendrograms
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(wine_norm,'complete'))

# Create Clusters (y)
hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters

y=pd.DataFrame(hclusters.fit_predict(wine_norm),columns=['clustersid'])
y['clustersid'].value_counts()

# Adding clusters to dataset
wine3=wine.copy()
wine3['clustersid']=hclusters.labels_
wine3

#############-----------2. K-Means Clustering
# Import Libraries
from sklearn.cluster import KMeans

# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(wine_norm)
    wcss.append(kmeans.inertia_)
# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#####---------Build Cluster algorithm using K=3
# Cluster algorithm using K=3
clusters3=KMeans(3,random_state=30).fit(wine_norm)
clusters3

clusters3.labels_

# Assign clusters to the data set
wine4=wine.copy()
wine4['clusters3id']=clusters3.labels_
wine4

wine4['clusters3id'].value_counts()
