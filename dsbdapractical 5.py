#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv('social_network_ads.csv')


# In[2]:


dt


# In[4]:


dt["Gender"].replace({"Male":1,"Female":0}, inplace=True)
dt


# In[5]:


dt=dt.drop(columns="User ID")


# In[6]:


dt.columns


# In[7]:


dt.describe()


# In[8]:


sns.pairplot(dt)
plt.show()


# In[9]:


x=dt[['Gender', 'Age', 'EstimatedSalary']]
y=dt['Purchased']


# In[10]:


y


# In[11]:


sns.displot(dt["Purchased"])


# In[12]:


corr_matrix=dt.corr()


# In[13]:


sns.heatmap(corr_matrix,annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[17]:


x_train


# In[18]:


print(y)


# In[19]:


y_train


# In[20]:


model=LogisticRegression()
model.fit(x_train,y_train)


# In[21]:


y_pred = model.predict(x_test)


# In[22]:


y_pred


# In[23]:


col = ["Age"]
dt.boxplot(col)


# In[24]:


model.score(x_train,y_train)


# In[25]:


model.score(x_test,y_test)


# In[26]:


c=sns.catplot(x="Gender",y="Purchased", data=dt,kind="bar",height=4)
c.set_ylabels("Prbability od purchase")


# In[27]:


ct = pd.crosstab(dt["Gender"],dt["Purchased"],normalize="index")
print(ct)


# In[28]:


ct.plot.bar(figsize=(6,4), stacked=True)
plt.show()


# In[29]:


from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, recall_score


# In[30]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[31]:


tn,fp,tp,fn=confusion_matrix(y_test,y_pred).ravel()


# In[32]:


a=accuracy_score(y_test,y_pred)    #tp+tn/total values
print("Accuracy score:",a)


# In[33]:


r=recall_score(y_test,y_pred)
print("Recall score:",r)


# In[35]:


e=1-a
print("Error rate:",e)


# In[ ]:




