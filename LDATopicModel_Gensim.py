# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
import en_core_sci_lg
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import math
import re
import string
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer= WordNetLemmatizer()
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

#import gensim related libraries
import gensim
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary
from gensim.models import Phrases
from collections import Counter
from gensim.models import Word2Vec

# read the cleaned and preprocessed file
cars_process = pd.read_csv("Cleaned_cars.csv")
cars_process.head()

#This function is used to get the part-of-speech(POS) for lemmatization
def get_pos_tag(tag):
    """This function is used to get the part-of-speech(POS) for lemmatization"""
    
    if tag.startswith('N') or tag.startswith('J'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN #default case
      
# Preprocessing the text by removing punctuations and stopwords, converting the text into lower case, lemmatizing the text before further analysis
def preprocess(text):
    punctuation= list(string.punctuation)
    doc_tokens= nltk.word_tokenize(text)
    word_tokens= [word.lower() for word in doc_tokens if not (word in punctuation or len(word)<=3)]
    # Lemmatize    
    pos_tags=nltk.pos_tag(word_tokens)
    doc_words=[wordnet_lemmatizer.lemmatize(word, pos=get_pos_tag(tag)) for word, tag in pos_tags]
    doc_words= [word for word in doc_words if word not in stopwords.words('english')]
    
    return doc_words#doc_tokens
  
cars_process['preprocess_text'] = cars_process['cleaned_complaint_body'].values.astype('U')
df_clean = cars_process['preprocess_text'].apply(preprocess)
df_clean.head()

# Tried multiple parts of speech and obtained best topic results using Nouns and verbs!
def get_nouns_adj(series):
    
    " Topic Modeling using only nouns and verbs"
    
    pos_tags= nltk.pos_tag(series)
    all_adj_nouns= [word for (word, tag) in pos_tags if (tag=="NN" or tag=="NNS" or tag=="VBD" or tag=="VBN")] 
    return all_adj_nouns

df_nouns_adj = df_clean.apply(get_nouns_adj)

# Creating bigrams from phrases
docs= list(df_nouns_adj)
bigram = gensim.models.Phrases(docs, min_count=10, threshold=20)
bigram_model = gensim.models.phrases.Phraser(bigram)

def make_ngrams(mod,texts):
    return [mod[doc] for doc in texts]

# Form Bigrams
data_words_bigrams = make_ngrams(bigram_model,docs)
# Checkout most frequent bigrams :
bigram_counter= Counter()
def print_freq_ngrams(phrases):
    for key in phrases.vocab.keys():
        if key not in stopwords.words('english'):
            if len(str(key).split('_'))>1:
                bigram_counter[key]+=phrases.vocab[key]

    for key, counts in bigram_counter.most_common(20):
        print(key,">>>>", counts)
    return
print('Bigrams')
print_freq_ngrams(bigram)

# Visualize the bigrams count
bigrams_count = pd.DataFrame(bigram_counter.most_common(20),
                             columns=['words', 'count'])
fig, axes = plt.subplots(figsize=(15, 30), dpi=100)
#print(bigram_counter)
# Plot horizontal bar graph
sns.barplot(y='words',x='count',data=bigrams_count)
plt.show()


#Create a dictionary and corpus for input to our LDA model. Filter out the most common and uncommon words.
dictionary= Dictionary(data_words_bigrams)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)
corpus = [dictionary.doc2bow(doc) for doc in docs]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

#Fit the corpus to the LDA model.  Using LDAMulticore model for quick and parallel fitting of the model 
from gensim.models.ldamulticore import LdaMulticore
passes= 150
np.random.seed(1) # setting up random seed to get the same results
ldamodel= LdaMulticore(corpus, 
                    id2word=dictionary, 
                    num_topics=12, 
                    chunksize= 4000, 
                    batch= True,
                    minimum_probability=0.001,
                    iterations=350,
                    passes=passes)  

# Shows the topics extracted after the text data is fit to the model
ldamodel.show_topics(num_words=25, formatted=False)

# Perform Clustering to group the topics into a cluster.  To apply clustering, the data should be converted to vectorized form
# perform TFIDF using the sklearn library
from sklearn.feature_extraction.text import TfidfVectorizer
max_features = 2**12
vectorizer = TfidfVectorizer(max_features=max_features,min_df=5, max_df=0.9, stop_words='english')
def vectorize(text, maxx_features,vectorizer):
    X = vectorizer.fit_transform(text.values.astype('U'))
    return X
  
processed_text = cars_process['cleaned_complaint_body']
processed_text.head()
X = vectorize(processed_text, max_features,vectorizer)
print(max_features)

#The dimension of vectorized form results in large number of features. So perform PCA to reduce the dimensionality
# perform PCA to reduce the number of features
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
X_reduced= pca.fit_transform(X.toarray())
X_reduced.shape

# perfom clustering using KMeans considering the euclidean distance between the data points
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

distortions = []
K = range(2, 20)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42)
    k_means.fit(X_reduced)
    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    print('Found distortion for {} clusters'.format(k))

# using the elbow-method, identify the optimal number of clusters to perform the final clustering on the data
X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

# Plot the elbow
plt.plot(K, distortions, 'b-')
plt.plot(X_line, Y_line, 'r')
plt.xlabel('No. of clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# After plotting using the elbow method, the optimal number of clusters obtained is 10.  Set n_clusters = 10
k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X_reduced)
cars_process['predicted_values'] = y_pred

# Plot the clusters
# sns settings
sns.set(rc={'figure.figsize':(11,9)})
# colors
palette = sns.hls_palette(10, l=.4, s=1.0)
sns.scatterplot(X_reduced[:,0], X_reduced[:,1], hue=y_pred, legend='full', palette=palette)
plt.title('Clustering Using K-Means')
plt.savefig("cluster_lab")
plt.show()

# Analyse the topics extracted and clusters. 

# Obtaining the main topic for each complaint
all_topics = ldamodel.get_document_topics(corpus)
num_docs = len(all_topics)

all_topics_csr= gensim.matutils.corpus2csc(all_topics)
all_topics_numpy= all_topics_csr.T.toarray()

imp_topics= [np.argmax(arr) for arr in all_topics_numpy]

# Visualize the topics count versus each cluster
sns.set(rc= {'figure.figsize': (10,8)})
sns.set_style('darkgrid')
plt.title('Number of topics vs Cluster')

cars_process.imp_topics.value_counts().plot(kind='bar')

# Group topics extracted, cluster-wise and print them
print(cars_process.groupby(['predicted_values'])['imp_topics'].value_counts(ascending=False, normalize=True))
cars_process['imp_topics']= imp_topics

# Impute the complaint severity of the issue into numerical form and rate its severity
cars_process.complaint_severity.unique()
cars_process['severity_rating'] = cars_process['complaint_severity'].apply(lambda score: 1 if (score == 'top most' or score == 'very urgent')
                                                                                   else 2 if (score == 'very high')
                                                                                   else 3 if (score == 'high')
                                                                                   else 4 if (score == 'moderate')
                                                                                   else 5 )
# To infer the actual insights of topics in each cluster, let us visualize each cluster against 4 categorical variables (['imp_topics','car_model','Brand','State'])

cat_cols= ['imp_topics','car_model','Brand','State']

cluster1= cars_process.loc[(cars_process.predicted_values==0)]
cluster2= cars_process.loc[(cars_process.predicted_values==1)]
cluster3= cars_process.loc[(cars_process.predicted_values==2)]
cluster4= cars_process.loc[(cars_process.predicted_values==3)]
cluster5= cars_process.loc[(cars_process.predicted_values==4)]
cluster6= cars_process.loc[(cars_process.predicted_values==5)]
cluster7= cars_process.loc[(cars_process.predicted_values==6)]
cluster8= cars_process.loc[(cars_process.predicted_values==7)]
cluster9= cars_process.loc[(cars_process.predicted_values==8)]
cluster10= cars_process.loc[(cars_process.predicted_values==9)]

#Visualizing the severity rating in each cluster (done for cluster 1)
sns.set(rc= {'figure.figsize': (8,6)})
pd.DataFrame((cluster1.severity_rating.value_counts()*100)/cars_process.severity_rating.value_counts()).plot(kind='bar')

# Visualizing all the categorical features for each cluster (done for cluster 1)
print('Visualizing categorical features:')
sns.set(rc= {'figure.figsize': (15,12)})
for i, col in enumerate(cat_cols):
    plt.figure(i)
    
# Similarly the above can be visualized for all the clusters.
# Insights from the above 
# Cluster 0 has complaints related to the following topics 
# •	Topic 6 highlights problems related to tyre, battery, warranty, purchase etc. 
# •	Topic 0 refers to certain service-related issues pertaining to time, repair, damage, bill
# •	Topic 4 has keywords which focuses on service-related complaints regarding Ford vehicle parts like clutch-plate, vehicle number-plate etc
# •	The severity of the complaint is “very urgent” or “top most”
# •	Most of the complaints in the cluster 0 are registered against the cars - Innova, Swift and Creta (115-125, totally)
# •	Maximum complaints in this cluster are registered against the car companies Hyundai, Maruti-Suzuki and Toyota (235-245, totally)
# •	Maharashtra has the maximum complaints (greater than 130)

#******************************************Thank You*************************************************

    chart= sns.countplot(y=cluster1[col], order= cluster1[col].value_counts().index)
