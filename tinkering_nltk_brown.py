
# coding: utf-8

# In[48]:


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


# In[49]:


#reading the csv first
speechData = pd.read_csv("/kaggle/input/speechdata - Sheet1.csv",encoding='latin-1')
speechData


# In[50]:


type(speechData)


# In[51]:


#checking the top data first
speechData_head=speechData.head(5)
speechData_head


# In[52]:


#checking the column in the data
for col in speechData.columns: 
    print(col) 


# In[55]:


#appears to be a space in between the two speechData columns
speechData.columns


# In[56]:


altered_speechData = speechData.rename(columns={'                                                                                                                                                                                      COMMENT':'Text','                 LABEL': 'Label'})


# In[57]:


#checking the columns again
altered_speechData.columns


# In[58]:


altered_speechData['Label'].nunique()


# In[59]:


altered_speechData['Label'].unique()


# In[60]:


altered_speechData_dict = {'Highly-Offensive':'Highly-Offensive','Moderate':'Neutral','Highly Offensive':'Highly-Offensive',
                          'Neutral':'Neutral','Highly -Offensive':'Highly-Offensive','Offensive':'Offensive',
                          'Hihly-Offensive':'Highly-Offensive'}
altered_speechData['Label']=altered_speechData['Label'].map(altered_speechData_dict)
altered_speechData.head(5)


# In[61]:


#checking whether the number of label values has been reduced to 3
altered_speechData['Label'].nunique()


# In[62]:


# import LabelEncoder 
from sklearn.preprocessing import LabelEncoder
# Instatniate LabelEncoder
le = LabelEncoder()
# LabelEncode Class column of df 
altered_speechData["Label_no"] = le.fit_transform(altered_speechData["Label"])
# Inspecting encoded df
altered_speechData.head()


# In[63]:


#checking whether the number of label_no values is also 3
altered_speechData['Label_no'].nunique()


# In[ ]:


#checking the unique values in the Label_no
altered_speechData['Label_no'].unique()

