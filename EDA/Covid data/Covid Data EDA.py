#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


df=pd.read_csv("covid.csv")


# In[4]:


df


# In[4]:


df.describe()


# In[5]:


df.duplicated().sum()


# In[6]:


df.isnull().sum()


# In[7]:


df.replace(np.nan,'0',inplace = True)
df.isnull().sum()


# In[11]:


df.dtypes


# In[10]:


df['ObservationDate']=pd.to_datetime(df['ObservationDate'])


# In[12]:


df.describe().transpose()


# In[13]:


df.head()


# In[14]:


df.tail()


# ##grouping cases as per the date

# In[17]:


date_wise=df.groupby(['ObservationDate']).agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum'})


# In[21]:


date_wise


# In[22]:


date_wise['Confirmed'].iloc[-1]


# ##total no of the active cases

# In[25]:


date_wise['Deaths'].iloc[-1]+date_wise['Recovered'].iloc[-1]


# In[28]:


plt.figure(figsize=(20,10))
sns.barplot(x=date_wise.index,y=date_wise['Confirmed']-date_wise['Recovered']-date_wise['Deaths'])
plt.xticks(rotation=90)
plt.title('Distribution for active cases')
plt.show()


# In[29]:


plt.figure(figsize=(20,10))
sns.barplot(x=date_wise.index,y=date_wise['Recovered']+date_wise['Deaths'])
plt.xticks(rotation=90)
plt.title('Distribution for closed cases')
plt.show()


# In[35]:


date_wise['week']=date_wise.index.weekofyear
date_wise


# In[37]:


week_num=[]
week_confirmed=[]
week_death=[]
week_recovered=[]
w=1
for i in list(date_wise['week'].unique()):
    week_confirmed.append(date_wise[date_wise['week']==i]['Confirmed'].iloc[-1])
    week_death.append(date_wise[date_wise['week']==i]['Deaths'].iloc[-1])
    week_recovered.append(date_wise[date_wise['week']==i]['Recovered'].iloc[-1])
    week_num.append(w)
    w=w+1


# In[42]:


plt.figure(figsize=(9,6))
plt.plot(week_num,week_confirmed,linewidth=3)
plt.plot(week_num,week_death)
plt.plot(week_num,week_recovered)
plt.xlabel('No of Week')
plt.ylabel('No of Cases')
plt.title('weekly progress of cases')
plt.show()


# In[64]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(21,7))
sns.barplot(x=week_num,y=pd.Series(week_confirmed).diff().fillna(0),ax=ax1)
sns.barplot(x=week_num,y=pd.Series(week_death).diff().fillna(0),ax=ax2)
sns.barplot(x=week_num,y=pd.Series(week_recovered).diff().fillna(0),ax=ax3)
ax1.set_xlabel('Number of week')
ax2.set_xlabel('Number of week')
ax3.set_xlabel('Number of week')

ax1.set_ylabel('Number of confrimed case')
ax2.set_ylabel('Number of death case')
ax3.set_ylabel('Number of recovered case')
plt.show()


# Averge increase in number of Confirmed cases everyday

# In[65]:


np.round(date_wise['Confirmed'].diff().fillna(0).mean())


# In[67]:


np.round(date_wise['Deaths'].diff().fillna(0).mean())


# In[68]:


np.round(date_wise['Recovered'].diff().fillna(0).mean())


# In[70]:


plt.figure(figsize=(11,5))
plt.plot(date_wise['Confirmed'].diff().fillna(0),label='Daily increase in Confirmed Case')
plt.plot(date_wise['Recovered'].diff().fillna(0),label='Daily increase in Recovered Case')
plt.plot(date_wise['Deaths'].diff().fillna(0),label='Daily increase in Death Case')
plt.xlabel('time')
plt.ylabel('nos of increase')
plt.show()


# In[72]:


country_wise=df[df['ObservationDate']==df['ObservationDate'].max()].groupby(['Country/Region'])
.agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum'})
.sort_values(['Confirmed'],ascending=False)
country_wise


# In[73]:


country_wise['mortality']=(country_wise['Deaths']/country_wise['Recovered'])*100


# In[74]:


country_wise


# In[75]:


country_wise['Recovered']=(country_wise['Recovered']/country_wise['Confirmed'])*100


# In[76]:


country_wise


# In[79]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,7))
top15_confirmed=country_wise.sort_values(['Confirmed'],ascending=False).head(15)
top15_death=country_wise.sort_values(['Deaths'],ascending=False).head(15)
sns.barplot(x=top15_confirmed['Confirmed'],y=top15_confirmed.index,ax=ax1)
sns.barplot(x=top15_death['Deaths'],y=top15_death.index,ax=ax2)
ax1.set_title('Top15 countries as per no of confirmed cases')
ax2.set_title('Top15 countries as per no of death cases')
plt.show()

