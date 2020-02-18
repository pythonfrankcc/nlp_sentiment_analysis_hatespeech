
# coding: utf-8

# In[79]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[80]:


import pandas as pd
ambazonia_textdata = pd.read_csv("../input/ambazonia.csv")
ambazonia_textdata.head(5)


# In[81]:


#lets look at the type of data that we have at hand
type(ambazonia_textdata)


# In[82]:


#lets see if it has the same error in naming as the SpeechData
ambazonia_textdata.columns


# In[83]:


#restructuring the dataset columns
ambazonia_textdata_altered =ambazonia_textdata.rename(columns={'  TIMESTAMP':'Timestamp', '                                                                                                                                                           TWEETS':'Tweets'})


# In[84]:


ambazonia_textdata_altered.head(5)


# In[85]:


#checking out if the columns have been altered
ambazonia_textdata_altered.columns


# In[86]:


#checking the type of data
type(ambazonia_textdata_altered)


# In[87]:


#lets work on preprocessing the Tweets part
tweets = ambazonia_textdata_altered.Tweets


# In[88]:


#checking out the just created Tweets dataset
tweets.head(5)


# In[89]:


#importing all necessary dependencies
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import seaborn

#preprocessing the tweets dataset that will be later pseudoencoded

## 1. Removal of punctuation and capitlization
## 2. Tokenizing
## 3. Removal of stopwords
## 4. Stemming

#based on what I have picked up I am going to use French stopwords
stopwords = nltk.corpus.stopwords.words("french")

#extending the stopwords to include other words used in twitter such as retweet(rt) etc.
other_exclusions = ["#ff", "ff", "rt","b"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()

def preprocess(tweets):  
    
    # removal of extra spaces
    regex_pat = re.compile(r'\s+')
    tweets_space = tweets.str.replace(regex_pat, ' ')

    # removal of @name[mention]
    regex_pat = re.compile(r'@[\w\-]+')
    tweets_name = tweets_space.str.replace(regex_pat, '')

    # removal of links[https://abc.com]
    giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweets = tweets_name.str.replace(giant_url_regex, '')
    
    # removal of punctuations and numbers
    punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
    # removal of capitalization
    tweets_lower = punc_remove.str.lower()
    
    # tokenizing
    tokenized_tweets = tweets_lower.apply(lambda x: x.split())
    
    # removal of stopwords
    tokenized_tweets=  tokenized_tweets.apply(lambda x: [item for item in x if item not in stopwords])
    
    # stemming of the tweets
    tokenized_tweets = tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x]) 
    
    for i in range(len(tokenized_tweets)):
        tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
        tweets_p= tokenized_tweets
    
    return tweets_p

processed_tweets = preprocess(tweets)

ambazonia_textdata_altered['processed_tweets'] = processed_tweets
ambazonia_textdata_altered.head(10)


# In[90]:


#lets look at whether there is an altering in the dataset type
type(ambazonia_textdata_altered)


# In[91]:


#lets see the exclusions that you missed from several rows so that you can get them out
ambazonia_textdata_altered.processed_tweets[0:3]


# In[92]:


ambazonia_textdata_altered.processed_tweets.iloc[2]


# In[93]:


#checking the context of the use of the word concordance
from nltk.text import Text
a = Text(word_tokenize(ambazonia_textdata_altered.processed_tweets.iloc[2]))
a.concordance('internet')

