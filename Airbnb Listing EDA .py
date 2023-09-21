#!/usr/bin/env python
# coding: utf-8

# # Full Data Analysis Case-study (Airbnb listings data for London)

# ### 1] Libraries & Data Importing

# ##### Import libraries and read in the listings csv file

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


df = pd.read_csv("listings.csv")


# ### 2] Data Understanding (Asking Questions)

# ##### Check the content of the DataFrame

# In[3]:


df.sample(2)


# ##### How many rows and columns are there?

# In[4]:


df.info()


# ### 3] Data Wrangling (Cleaning & Manipulation)

# ##### Drop unnecessary columns

# In[5]:


# df.drop(["id", "name", "host_id", "host_name", "neighbourhood_group", "number_of_reviews", "last_review", "reviews_per_month", "number_of_reviews_ltm", "license"], axis=1, inplace=True)


# In[6]:


df=df[["neighbourhood","latitude","longitude","price","minimum_nights","availability_365","room_type","calculated_host_listings_count"]]


# #### Examining Changes

# In[7]:


df.head()


# In[8]:


df.info()


# ##### Are there any missing / duplicated data?

# In[9]:


df.duplicated().value_counts()


# In[10]:


df = df.drop_duplicates()


# In[11]:


df.isna().sum()


# ### 4] Data Analysis & Visualization (EDA & Statistical Analysis)

# #### Examine Continous Variables

# In[12]:


df.describe()


# In[13]:


df.describe(include=['object', 'bool'])


# #### Get Correlation between different variables

# In[14]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)


# #### Multivariate visualization

# In[15]:


sns.pairplot(df)


# ######################################################################

# #### Neighbourhood

# In[16]:


df["neighbourhood"].value_counts()


# In[17]:


df["neighbourhood"].unique()


# In[18]:


plt.figure(figsize=(15,10))
sns.countplot(x='neighbourhood', data=df)
plt.grid()
plt.show()


# #### Room type

# In[19]:


df["room_type"].value_counts()


# In[20]:


df["room_type"].unique()


# In[21]:


plt.figure(figsize=(10,7))
sns.countplot(x='room_type', data=df)
plt.grid()
plt.show()


# #### calculated host listings count

# In[22]:


df["calculated_host_listings_count"].value_counts()


# In[23]:


df["calculated_host_listings_count"].unique()


# In[24]:


plt.figure(figsize=(15,7))
sns.boxplot(df['calculated_host_listings_count'])


# #############################################################

# #### The relation between price and  calculated host listings count 

# In[25]:


sns.jointplot(x='calculated_host_listings_count', y='price', data=df, kind='scatter')


# #### The relation between price,  calculated host listings count and room type

# In[26]:


sns.lmplot(x='calculated_host_listings_count', y='price', data=df, hue='room_type', fit_reg=False)


# #### The relation between price and  availability of room 

# In[27]:


sns.jointplot(x='availability_365', y='price', data=df, kind='scatter')


# #### The relation between price, availability of room and room type

# In[28]:


sns.lmplot(x='availability_365', y='price', data=df, hue='room_type', fit_reg=False)


# #### The relation between minimum nights and calculated host listings count

# In[29]:


sns.jointplot(x='calculated_host_listings_count', y='minimum_nights', data=df, kind='scatter')


# #### The relation between minimum nights, calculated host listings count and room type

# In[30]:


sns.lmplot(x='calculated_host_listings_count', y='minimum_nights', data=df, hue='room_type', fit_reg=False)


# #### The relation between minimum nights and availability of room 

# In[31]:


sns.jointplot(x='minimum_nights', y='availability_365', data=df, kind='scatter')


# #### The relation between minimum nights, availability of room and room type

# In[32]:


sns.lmplot(x='minimum_nights', y='availability_365', data=df, hue='room_type', fit_reg=False)


# #################################################################

# #### Map of Neighbourhood

# In[33]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood)
plt.ioff()


# #### Map of Room type

# In[34]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.room_type)
plt.ioff()


# #### Map of Availability of Room

# In[35]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.availability_365)
plt.ioff()


# #### Map of calculated host listings count

# In[36]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.calculated_host_listings_count)
plt.ioff()


# #### Map of price

# In[37]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.price)
plt.ioff()


# #### Map of minimum nights

# In[38]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.minimum_nights)
plt.ioff()

