#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import warnings


# In[6]:


d=pd.read_excel('super.xls')
d


# In[7]:


d.describe


# In[8]:


d.drop(['Row ID','Postal Code'],axis=1,inplace=True)


# In[9]:


d


# In[10]:


d['Sales'].mean()


# In[11]:


d['Sales'].count()


# In[12]:


d['Sales'].min()


# In[13]:


d['Sales'].max()


# In[32]:


d['Category'].unique()


# In[14]:


d1=d.copy()


# In[15]:


d1


# In[35]:


d1=d1.pivot_table('Sales',columns='Region',aggfunc='sum')
d1


# In[16]:


d2=d.copy()
d2=d2.pivot_table('Profit',columns='Region',aggfunc='sum')
d2


# In[2]:


d3=d.copy()
d3=d3.groupby(['Category'])['Sales'].sum()
d3


# In[18]:


rport=d.copy()
rport=pd.DataFrame(d.groupby(['Ship Mode','Segment','Category','Sub-Category','State','Region'])['Sales','Quantity','Discount','Profit'].sum().reset_index())
rport


# In[19]:


state_profit_d=pd.pivot_table(data=d[['State','Profit']],
                              index=['State'],
                              values='Profit',
                              aggfunc='max'
                             )
state_profit_d.sort_values(by='Profit',ascending=False).head(10)


# In[20]:


d_cat=pd.pivot_table(data=d[['Category','Sub-Category','Sales']],
                              index=['Category','Sub-Category'],
                              values='Sales',
                              aggfunc='max'
                             )
d_cat


# In[21]:


d_cat_q=pd.pivot_table(data=d[['Category','Sub-Category','Quantity']],
                              index=['Category','Sub-Category'],
                              values='Quantity',
                              aggfunc='sum'
                             )
d_cat_q


# In[22]:


plt.bar(d['Quantity'],d['Profit'])
plt.title('Profit vs Quantity', fontsize=15,color='red')
plt.xlabel('Quantity',fontsize=15)
plt.ylabel('Profit',fontsize=15)
plt.grid(True)
plt.show()


# In[23]:


plt.figure(figsize=(18,5))
plt.bar('Sub-Category','Category',data=d,color='green')
plt.show()


# In[25]:


print(d['State'].value_counts())
plt.figure(figsize=(10,4))
sns.countplot(x=d['State'])
plt.xticks(rotation=90)
plt.show()


# In[26]:


print(d['Sub-Category'].value_counts())
plt.figure(figsize=(10,6))
sns.countplot(x=d['Sub-Category'])
plt.xticks(rotation=90)
plt.show()


# In[27]:


plt.bar(d['Quantity'],d['Sales'],color="red")
plt.title('Sales Vs Quantity',fontsize=15,color='blue')
plt.xlabel('Quantity',fontsize=15)
plt.ylabel('Sales',fontsize=15)
plt.grid(True)
plt.show()


# In[30]:


sns.relplot(x='Quantity',y='Profit',hue='Segment',data=d)
plt.show()


# In[31]:


sns.countplot(x='Segment',data=d,palette='coolwarm')
plt.show()


# In[38]:


plt.title('Segment',fontsize=20)
d['Segment'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)
plt.show()


# In[33]:


sns.countplot(x='Category',data=d,palette='tab10')
plt.show()


# In[34]:


d.groupby('Segment')[['Profit','Sales']].sum().plot.bar(color=['red','blue'],figsize=(8,5))
plt.ylabel('Profit/Loss and Sales')
plt.show()


# In[37]:


plt.title('Region',fontsize=20)
d['Region'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)
plt.show()

