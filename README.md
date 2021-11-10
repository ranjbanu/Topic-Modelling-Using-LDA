# Topic-Modelling-Using-LDA
This repository contains the code to obtain insights from the textual data.The corpus has been mined using various text processing techniques such that it can be fed into NLP to get meaningful insights. 

# Objective
The objective of the project was to extract data from the consumer complaint website, analyze the data using NLP techniques and derive insights that could be used by the potential stakeholders

# Text Preprocessing
Text cleaning and feature extraction
Libraries: pandas, numpy, regex, spacy, texthero, nltk and gensim

* Consider the complaint text, identify missing values in the complaint text field and drop those rows
* Extract city and state from the location data
* Format the date column
* Extract the brand / seller name from the complaint title
* Extract new features – Brand and Model name of the automobile from the complaint text
* Add another feature, complaint_severity which rates the complaint severity. Tag this value as high, moderate, low based on the severity of the sentiment in the complaint. 
* Clean the complaint text by removing occurrences of links, extra spaces, new line characters and special characters. 
* Remove all the stopwords in English from the complaint text including the open salutations.
* Convert all the contractions such as “didn’t”, “what’s” to “did not” , “what is”.
* Remove customized stopwords and Hinglish words such as “madam”, “sir”, “par”,”gaya”, “hai” etc

# Topic modelling using LDA
* Text pre-processing is an essential step before LDA modelling.  Stop-words, punctuations and special characters should be eliminated to reduce overhead to the model. Lemmatization helps reduce the words to the root level.  
**Note**: LDA modelling begins with random assignment of topics to each word and iteratively improves the assignment of topics to words.
* Create a dictionary using bigram word tokens.  Using the doc2bow (document to bag of words) function from the dictionary library, create a corpus of text which will serve as an input to the LDA model.
* Hyper parameters in LDA modelling:
	* Alpha: represents document-topic density i.e. the number of topics per document. Low value of alpha shows fewer number of topics in the document mix, while higher value    
    indicates that the documents have a greater number of topics in the mix.
	* Beta: represents topic-word density i.e the number of words per topic.  Lower the value of beta, fewer are the number of words in the topic and vice-versa.
	* K = No of topics to be taken into account.

# Clustering using K-Means
Clustering is an unsupervised learning problem. The topics extracted will be naturally grouped under clusters in the feature space of the text data. One cluster will have many topics under it.  The text data must be converted into a vectorized form before fitting it to an unsupervised machine model.  Vectorize the text data using TFIDFVectorizer.   Since the vectorized matrix contains numerous features, reduce the dimensionality by applying PCA.  

K-Means is the most popular clustering algorithm used to discover interesting patterns in the data.  Apply K-Means with a range of clusters between 2 and 20.  Then using the elbow method, identify the optimal number of clusters for the text data. 

# Inference from Clustering against Topic Modelling 
Cluster 0 has complaints related to the following topics 
* Topic 6 highlights problems related to tyre, battery, warranty, purchase etc. 
* Topic 0 refers to certain service-related issues pertaining to time, repair, damage, bill
* Topic 4 has keywords which focuses on service-related complaints regarding Ford vehicle parts like clutch-plate, vehicle number-plate etc
* The severity of the complaint is “very urgent” or “top most”
* Most of the complaints in the cluster 0 are registered against the cars - Innova, Swift and Creta (115-125, totally)
* Maximum complaints in this cluster are registered against the car companies Hyundai, Maruti-Suzuki and Toyota (235-245, totally)
* Maharashtra has the maximum complaints (greater than 130).

**Similarly, we can infer and summarize for other clusters too.**


# Conclusion
LDA Topic modelling on a structured text data helps us identify the pattern of the text and extract relevant topics after analysis




