
# coding: utf-8

# In[134]:


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


# In[135]:


#reading the csv first
speechData = pd.read_csv("/kaggle/input/speechdata - Sheet1.csv",encoding='latin-1')
speechData


# In[136]:


type(speechData)


# In[137]:


#checking the top data first
speechData_head=speechData.head(5)
speechData_head


# In[138]:


#checking the column in the data
for col in speechData.columns: 
    print(col) 


# In[139]:


#appears to be a space in between the two speechData columns
speechData.columns


# In[140]:


altered_speechData = speechData.rename(columns={'                                                                                                                                                                                      COMMENT':'Text','                 LABEL': 'Label'})


# In[141]:


#checking the columns again
altered_speechData.columns


# In[142]:


#looking at the number of unique labels
altered_speechData['Label'].nunique()


# In[143]:


#what are those labels
altered_speechData['Label'].unique()


# In[144]:


#altering the labels so that there are only 3 unique labels
altered_speechData_dict = {'Highly-Offensive':'Highly-Offensive','Moderate':'Neutral','Highly Offensive':'Highly-Offensive',
                          'Neutral':'Neutral','Highly -Offensive':'Highly-Offensive','Offensive':'Offensive',
                          'Hihly-Offensive':'Highly-Offensive'}
altered_speechData['Label']=altered_speechData['Label'].map(altered_speechData_dict)
altered_speechData.head(5)


# In[145]:


#checking whether the number of label values has been reduced to 3
altered_speechData['Label'].nunique()


# In[146]:


#adding a new column of label_no that represents labels
# import LabelEncoder 
from sklearn.preprocessing import LabelEncoder
# Instatniate LabelEncoder
le = LabelEncoder()
# LabelEncode Class column of df 
altered_speechData["Label_no"] = le.fit_transform(altered_speechData["Label"])
# Inspecting encoded df
altered_speechData.head()


# In[147]:


#checking whether the number of label_no values is also 3
altered_speechData['Label_no'].nunique()


# In[148]:


#checking the unique values in the Label_no
altered_speechData['Label_no'].unique()


# In[149]:


#checking the tail to see whether their is randomness in the label column
altered_speechData.tail(5)


# In[150]:


#lets shuffle the data
shuffled_altered_speechData = altered_speechData.sample(frac=1).reset_index(drop=True)
shuffled_altered_speechData.tail(5)


# In[151]:


#checking the distribustion of the labels to see the most rampant
shuffled_altered_speechData['Label'].hist()


# In[152]:


#looking at the above histogram Neutral is the most prevalent then Highly-Offensive then Offensive
# collecting only the Text from the csv file into a variable name comments_dataset
comments_dataset=shuffled_altered_speechData.Text


# In[153]:


#checking out the data that has been picked out
comments_dataset.head()


# In[154]:


#checking the comments_dataset type
type(comments_dataset)


# In[155]:


#importing some necessary dependencies
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import seaborn

## 2. Tokenizing
## 3. Removal of stopwords
## 4. Stemming

stopwords = nltk.corpus.stopwords.words("english")

#extending the stopwords to include other words used in twitter such as retweet(rt) etc.
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()

def preprocess(comments_dataset):  
    
    # removal of extra spaces
    regex_pat = re.compile(r'\s+')
    comments_dataset_space = comments_dataset.str.replace(regex_pat, ' ')

    # removal of @name[mention]
    regex_pat = re.compile(r'@[\w\-]+')
    comments_dataset_name = comments_dataset_space.str.replace(regex_pat, '')

    # removal of links[https://abc.com]
    giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    comments_dataset = comments_dataset_name.str.replace(giant_url_regex, '')
    # removal of punctuations and numbers
    punc_remove = comments_dataset.str.replace("[^a-zA-Z]", " ")
    # removal of capitalization
    comments_dataset_lower = punc_remove.str.lower()
    
    # tokenizing
    tokenized_comments_dataset = comments_dataset_lower.apply(lambda x: x.split())
    
    # removal of stopwords
    tokenized_comments_dataset=  tokenized_comments_dataset.apply(lambda x: [item for item in x if item not in stopwords])
    
    # stemming of the tweets
    tokenized_comments_dataset = tokenized_comments_dataset.apply(lambda x: [stemmer.stem(i) for i in x]) 
    
    for i in range(len(tokenized_comments_dataset)):
        tokenized_comments_dataset[i] = ' '.join(tokenized_comments_dataset[i])
        comments_dataset_p= tokenized_comments_dataset
    
    return comments_dataset_p

processed_comments_dataset = preprocess(comments_dataset)   

shuffled_altered_speechData['processed_Text'] = processed_comments_dataset
shuffled_altered_speechData


# In[156]:


#checking the type after the addition of the processed text data type
type(shuffled_altered_speechData)


# In[157]:


#importing necessary dependencies for visualization
#from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')

# visualizing which of the word is most commonly used in the processed_Text

import matplotlib.pyplot as plt
from wordcloud import WordCloud

all_words = ' '.join([text for text in shuffled_altered_speechData['processed_Text'] ])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[162]:


#Feauture Generation
# Bigram Features

bigram_vectorizer = CountVectorizer(ngram_range=(1,2),max_df=0.75, min_df=1, max_features=10000)
# bigram feature matrix
bigram = bigram_vectorizer.fit_transform(processed_comments_dataset).toarray()
bigram


# In[163]:


#TF-IDF Features

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)

# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(shuffled_altered_speechData['processed_Text'] )
tfidf


# In[168]:


#Building a model using Logistic Regression
# Using Bigram Features
X = pd.DataFrame(bigram)
y = shuffled_altered_speechData['Label_no'].astype(int)
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)


model = LogisticRegression(class_weight='balanced',penalty="l2", C=0.01).fit(X_train_bow,y_train)
y_preds = model.predict(X_test_bow)
report = classification_report( y_test, y_preds )
print(report)

print("Accuracy Score:" , accuracy_score(y_test,y_preds))


# In[ ]:


# Running the model Using TFIDF without additional features

X = tfidf
y = shuffled_altered_speechData['Label_no'].astype(int)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)


model = LogisticRegression().fit(X_train_tfidf,y_train)
y_preds = model.predict(X_test_tfidf)
report = classification_report( y_test, y_preds )
print(report)

print("Accuracy Score:" , accuracy_score(y_test,y_preds))

#this is an expression of the best model the next we are going to be looking at is concordance

