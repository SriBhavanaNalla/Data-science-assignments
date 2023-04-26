# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#pip install mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
# Loading Dataset
data = pd.read_csv('book.csv')

#EDA
data.head()

# checking the random entries in the data

data.sample(10)
data.shape
data.info()

# Word cloud of most frequent books
#pip install wordcloud
from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = (13, 13)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(data))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Items',fontsize = 20)
plt.show()

book_count = []
col_names = data.columns
for col_name in col_names:
    book_count.append(data[col_name].value_counts()[1])
    
plt.bar(col_names, book_count)

#Apriori Algorithm for min_support = 0.1
frequent_itemsets1 = apriori(data, min_support=0.1, use_colnames=True)
frequent_itemsets1 

frequent_itemsets1 = apriori(data, min_support = 0.1, use_colnames=True)
frequent_itemsets1['length'] = frequent_itemsets1['itemsets'].apply(lambda x: len(x))
frequent_itemsets1

#Rules when min_support = 0.1 and min_threshold for lift is 0.5
rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=0.5)
rules1
rules1.sort_values('lift',ascending = False)

plt.scatter(rules1['support'], rules1['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

#Rules when min_support = 0.1 and min_threshold for confidence is 0.5
rules2 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.5)
rules2
rules2 = rules2.sort_values(['confidence', 'lift'], ascending =[False, False])
rules2

plt.scatter(rules2['support'], rules2['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

#Apriori Algorithm for min_support = 0.2
frequent_itemsets2 = apriori(data, min_support=0.2, use_colnames=True)
frequent_itemsets2
frequent_itemsets2 = apriori(data, min_support = 0.2, use_colnames=True)
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))
frequent_itemsets2

#Rules when min_support = 0.2 and min_threshold for lift is 0.5
rules3 = association_rules(frequent_itemsets2, metric="lift", min_threshold=0.1)
rules3
plt.scatter(rules3['support'], rules3['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

#Rules when min_support = 0.1 and min_threshold for confidence is 0.5
rules4 = association_rules(frequent_itemsets2, metric="confidence", min_threshold=0.5)
rules4
rules4 = rules4.sort_values(['confidence', 'lift'], ascending =[False, False])
rules4
plt.scatter(rules4['support'], rules4['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()