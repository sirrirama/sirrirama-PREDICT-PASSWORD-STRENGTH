#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[8]:


data= pd.read_csv(r"C:\Users\sriram kumar\Dropbox\PC\Downloads\data(1).csv",error_bad_lines= False)
data.head()


# In[9]:


data['strength'].unique()


# In[10]:


data.isna().sum()


# In[11]:


data[data['password'].isnull()]


# In[12]:


data.dropna(inplace= True)


# In[13]:


data.isna().sum()


# In[14]:


sns.countplot(data['strength'])


# # data analysis

# The above data is inclined towards strength= 1, i.e. if we use this data as it is then our model will predict almost all the test results to have strength= 1
# So we will do certain manipulation in this dataset

# In[16]:


password_tuple= np.array(data)


# In[17]:


password_tuple


# # Now we will shuffle our array

# In[18]:


import random
random.shuffle(password_tuple)


# # Independent Data in dataset

# In[19]:


x= [label[0] for label in password_tuple]
x


# # Dependent data in dataset

# In[21]:


y= [labels[1] for labels in password_tuple]
y


# # Applying TF-IDF

# The data should be in character datatype

# # Converting the data into character format

# In[22]:


def convert_to_char(input):
    character=[]
    for i in input:
        character.append(i)
    return character


# In[23]:


convert_to_char('kzde5577')


# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[25]:


vectorizer= TfidfVectorizer(tokenizer=convert_to_char)


# In[26]:


X=vectorizer.fit_transform(x)


# In[27]:


X.shape


# In[28]:


vectorizer.get_feature_names()


# # To get the data having its own importance

# In[29]:


first_document_vector= X[0]


# In[30]:


first_document_vector.T.todense()


# In[31]:


df= pd.DataFrame(first_document_vector.T.todense(), index= vectorizer.get_feature_names(), columns=['TF-IDF'])
df.sort_values('TF-IDF', ascending= False)


# # Applying Logistic Regression

# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.2)


# In[34]:


x_train.shape


# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


clf= LogisticRegression(random_state= 0, multi_class= 'multinomial')


# In[37]:


clf.fit(x_train, y_train)


# In[38]:


y_pred= clf.predict(x_test)
y_pred


# # To Determine accuracy:-

# Import Confusion Matrix,
# Accuracy Score,
# Classification Report

# In[40]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[41]:


cm= confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


# In[42]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Completed

# For better accuracy, hypertuning can be done.
