#!/usr/bin/env python
# coding: utf-8

# # Scrape Flipkart Data using Python

# <b>Step 1: Install/ Import the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import bs4
from bs4 import BeautifulSoup as bs
import requests


# <b>Step 2: Choose the data you need to extract

# In[2]:


link='https://www.flipkart.com/search?q=samsung+mobiles&sid=tyy%2C4io&as=on&as-show=on&otracker=AS_QueryStore_HistoryAutoSuggest_0_2&otracker1=AS_QueryStore_HistoryAutoSuggest_0_2&as-pos=0&as-type=HISTORY&as-searchtext=sa'


# <b>Step 3: Send HTTP request to the URL of the page you want to scrape

# We will use the Request library to send the request HTTP request to the URL of the page mentioned in the above code and store the response in the page variable. It will give us the HTML content of the page which is used to scrape the required data from the page.

# In[3]:


page = requests.get(link)


# <b>Print The Content Of The Page

# In[4]:


page.content


# Now that we have got the HTML content of the page, we need to parse the data and store it in soup variable to structure it into a different format which will help in data extraction.

# In[5]:


soup = bs(page.content, 'lxml')
print(soup.prettify())


# In[6]:


soup.title


# In[7]:


soup.div


# In[8]:


soup.title.string


# <b> Step 4: Inspect the page and write codes for extraction

# In this step, we will Inspect the Flipkart page from where we need to extract the data and read the HTML tags. To do the same, We can right-click on the page and click on “Inspect”.
# As we click on “Inspect” the following screen will appear where all the HTML tags used are present which will help us to extract the exact data we need.
# 

# <b>Extracting the Name of the Product
# 

# In[9]:


name=soup.find('div',class_="_4rR01T")
print(name)


# In[10]:


name.text


# <b>Extracting the Price of the Product

# In[11]:


price=soup.find('div',class_='_30jeq3 _1_WHN1')
print(price)
price.text


# <b>Extracting the Rating of the Product

# In[12]:


rating=soup.find('div',class_="_3LWZlK")
print(rating)
rating.text


# <b>Find All the Links of the product

# In[13]:


all_links=soup.find_all("a")
for i in all_links:
    print(i.get("href"))


# In[14]:


all_table=soup.find_all("table_tab")
print(all_table)


# <b>Extracting the Name of the All Product In A Page

# In[15]:


product=soup.find_all('div',{'class':'_4rR01T'})
print(product)


# <b>Extracting the Price of the All Product In A Page

# In[16]:


price_list=soup.find_all('div',{'class':'_30jeq3'})
print(price_list)


# <b>Extracting the Ratings of the All Product In A Page

# In[17]:


rating_list=soup.find_all('div',{'class':'_3LWZlK'})
print(rating_list)


# <b>Define the list

# In[18]:


product_name=[]              #List to store the name of the product
prices=[]                #List to store price of the product
ratings=[]               #List to store rating of the product


# <b>Append the Product Name in Product list

# In[19]:


for i in product:
    p=i.text
    product_name.append(p)


# In[20]:


print(product_name)


# <b>Append the Price in Price list

# In[21]:


for i in price_list:
    p=i.text
    prices.append(p)


# In[22]:


prices


# <b>Append the Ratings in Rating list

# In[23]:


for i in rating_list:
    p=i.text
    ratings.append(p)


# In[24]:


ratings


# <b>length of list

# In[25]:


print(len(product_name))
print(len(ratings))
print(len(prices))


# <b>All arrays must be of the same length

# In[27]:


prices=prices[0:24]
ratings=ratings[0:24]
print(len(ratings))
print(len(prices))


# <b>Storing the data into the structured format in the Data Frame.

# In[28]:


import pandas as pd
df=pd.DataFrame({'Product Name':product_name,'Price':prices,'Rating':ratings})
df.head(10)

