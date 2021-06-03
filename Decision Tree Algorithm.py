#!/usr/bin/env python
# coding: utf-8

# # PREDICTION USING DECISION TREE ALGORITHM
# 

# #  Author = Shantanu.Kakkara

# In[23]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.metrics as sm
import matplotlib.pyplot as mt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# In[24]:


#data reading
data = pd.read_csv("C:/Users/shant/OneDrive/Desktop/Iris.csv", index_col = 0)
data.head()


# In[25]:


data.info()


# In[26]:


data. describe()


# # Now we will input Data Visualization

# In[27]:


sns.pairplot( data, hue = 'Species')


# # Now we will find the correlation between the Matrix

# In[29]:


data.corr()


# In[30]:


sns.heatmap( data.corr())


# In[31]:


target = data['Species']
df = data.copy()
df=df.drop('Species', axis=1)
df.shape


# In[32]:


X = data.iloc[:, [0,1,2,3]].values
le=LabelEncoder()
data['Species']=le.fit_transform(data['Species'])
y=data['Species'].values
data.shape


# # Model Training

# In[33]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training Split:",X_train.shape)
print("Testing Split:",X_test.shape)


# In[34]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print("Decision Tree Classifier Created")


# # Classification Report and Confusion Matrix

# In[35]:


y_pred = dtree.predict(X_test)
print("Classification Report:\n",classification_report(y_test,y_pred))


# In[36]:


print("Accuracy:",sm.accuracy_score(y_test,y_pred))


# In[37]:


#confusion matrix
cm=confusion_matrix(y_test, y_pred)
cm


# In[38]:


#visualizing the graph
mt.figure(figsize = (20,10))
tree= plot_tree(dtree,feature_names = df.columns, precision = 2, rounded = True , filled = True, class_names = target.values)


# In[ ]:




