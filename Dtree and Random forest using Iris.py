#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import sklearn as sk


# In[5]:


dataset=pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")


# In[6]:


x=dataset.iloc[:,1:4].values


# In[7]:


print(dataset)


# In[8]:


y=dataset.iloc[:,4].values


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


y=LabelEncoder().fit_transform(y)


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


classifier_dtree=DecisionTreeClassifier(criterion='entropy')
classifier_dtree.fit(x_train,y_train)


# In[15]:


y_pred=classifier_dtree.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[16]:


from sklearn.ensemble import RandomForestClassifier


# In[17]:


classifier_random_forest=RandomForestClassifier()
classifier_random_forest.fit(x_train,y_train)


# In[18]:


y_pred=classifier_random_forest.predict(x_test)


# In[19]:


confusion_matrix(y_test,y_pred)


# In[20]:


import pickle


# In[21]:


pickle.dump(classifier_random_forest,open('model.pkl','wb'))


# In[22]:


model=pickle.load(open('model.pkl','rb'))


# In[36]:


ram=dataset.iloc[1:2,1:4].values


# In[39]:


print(model.predict([(2,3,5),(2,6,7)]))


# In[ ]:




