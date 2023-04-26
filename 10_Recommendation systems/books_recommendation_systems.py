import pandas as pd
import numpy as np
df=pd.read_csv('books.csv', encoding='latin-1')
df.head()

df.rename(columns={"User.ID":"user_id","Book.Title":"book_title","Book.Rating":"book_rating"},inplace=True)
df.info()

df.isnull().sum()

df.duplicated().sum()
book=df.iloc[:,1:]
book.head(3)

len(book.user_id.unique())

len(book.book_title.unique())
user_book = book.pivot_table(index='user_id',columns='book_title',values='book_rating')
user_book.head()

#Impute those NaNs with 0 values
user_book.fillna(0, inplace=True)
user_book.head()

#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
user_sim = 1 - pairwise_distances( user_book.values,metric='cosine')
user_sim


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)
#Set the index and column names to user ids 
user_sim_df.index = book.user_id.unique()
user_sim_df.columns = book.user_id.unique()
user_sim_df.iloc[0:5, 0:5]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

#5 Similar Users
user_sim_df.idxmax(axis=1)[0:5]