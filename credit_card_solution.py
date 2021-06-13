#!/usr/bin/env python
# coding: utf-8

# ### Credit Card Lead Prediction
# #### Problem Statement
# - Defining the problem statement
# - Collecting the data
# - Exploratory data analysis
# - Feature engineering
# - Modelling
# - Testing

# #### 1. Defining the problem statement
# Happy Customer Bank is a mid-sized private bank wants to cross sell its credit cards to its existing customers and is looking for help in identifying customers that could show higher intent towards a recommended credit card.

# #### 2. Collecting the data
# Using pandas to load train and test csv files
# 

# In[1]:


import pandas as pd
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# #### 3. Exploratory Data Analysis
# Printing the first five rows of the train data

# In[2]:


train.head()


# In[3]:


test.head()


# ##### Train Set Data Attributes
# - Variable: Definition : Definition
# - ID                   : Unique Identifier for a row 	
# - Gender               : Gender of the Customer 	
# - Age                  : Age of the Customer (in Years)
# - Region_Code          : Code of the Region for the customers
# - Occupation           : Occupation Type for the customer 
# - Channel_Code         : Acquisition Channel Code for the Customer  (Encoded)
# - Vintage              : Vintage for the Customer (In Months)
# - Credit_Product       : If the Customer has any active credit product (Home loan,Personal loan, Credit Card etc.)
# - Avg_Account_Balance  : Average Account Balance for the Customer in last 12 Months              
# - Is_Active            : If the Customer is Active in last 3 Months
# - Is_Lead(Target)      : If the Customer is interested for the Credit Card, 0 : Customer is not interested, 1 : Customer is interested

# In[4]:


print("train set rows\t\t: {}".format(train.shape[0]))
print("train set columns\t: {}\n".format(train.shape[1]))
print("test set rows\t\t: {}".format(test.shape[0]))
print("test set columns\t: {}".format(test.shape[1]))


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


train.isnull().sum()


# In[8]:


test.isnull().sum()


# Both train and test set have NaN values under Credit Product attribute.

# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# In[10]:


def bar_chart(feature):
    lead = train[train['Is_Lead']==1][feature].value_counts()
    not_lead = train[train['Is_Lead']==0][feature].value_counts()
    df = pd.DataFrame([lead,not_lead])
    df.index = ['Is a lead','Not a lead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[11]:


bar_chart('Gender')


# The above chart suggests that genders really doesn't define if someone is not interested in credit cards. Among those who are interested to have credit card there are more men than woman.

# In[12]:


bar_chart('Channel_Code')


# In[13]:


bar_chart('Is_Active')


# In[14]:


bar_chart('Credit_Product')


# In[15]:


bar_chart('Occupation')


# #### 4. Feature Engineering
# The algorithms in machine learning require a numerical representation of data so that
# such representations facilitate processing and statistical analysis. 
# 
# The following attributes can be looked upon for feature engineering
# - Gender *
# - Age *
# - Region Code
# - Occupation *
# - Channel Code *
# - Vintage *
# - Credit Product 
# - Average Account Balance *
# - Is Active *
# 
# Since Gender, Occupation, Channel Code, and Is Active attributes have only few dimentions and do not have Nan values, it would be easy to map out their values to numrica values easily.

# #### 4.1 Gender
# Gender mapping [male:0, female: 1]

# In[16]:


train_test_data = [train, test]


# In[17]:


gender_mapping = {"Male": 0, "Female": 1}
for dataset in train_test_data:
    dataset['Gender'] = dataset['Gender'].map(gender_mapping)


# #### 4.2 Occupation
# Occupation Mapping [Self Employed: 0, other: 1, Salaried: 2, Entreprenuer: 3]

# In[18]:


occupation_mapping = {"Self_Employed": 0, "Other": 1, "Salaried": 2, "Entrepreneur": 3}
for dataset in train_test_data:
    dataset['Occupation'] = dataset['Occupation'].map(occupation_mapping)


# #### 4.3 Channel Code
# Channel code Mapping [X1: 0, X2: 1, X3: 2, X4: 3]

# In[19]:


channel_code_mapping = {"X1": 0, "X2": 1, "X3": 2, "X4": 3}
for dataset in train_test_data:
    dataset['Channel_Code'] = dataset['Channel_Code'].map(channel_code_mapping)


# #### 4.4 Is Active
# Credit Product Mapping [No: 0, Yes: 1]

# In[20]:


is_active_mapping = {"No": 0, "Yes": 1}
for dataset in train_test_data:
    dataset['Is_Active'] = dataset['Is_Active'].map(is_active_mapping)


# #### 4.5 Age

# In[21]:


print("Maximum Age: {}".format(train["Age"].max()))
print("Minimum Age: {}".format(train["Age"].min()))


# In[22]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show() 


# According the the age distribution with respect to the "Is Lead" attribute, there are several cross over points.
# 

# In[23]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(20, 40)


# In[24]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(35, 70)


# In[25]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(70, 85)


# #### 4.5.1 Binning - Age
# Binning/Converting Numerical Age to Categorical Variable
# 
# Feature vector map
# 
# Under 37 years old:           0
# 
# More the 37 and less than 70: 1
# 
# Greater than 70:              2
# 

# In[26]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 37, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 37) & (dataset['Age'] <= 70), 'Age'] = 1
    dataset.loc[ dataset['Age'] > 70, 'Age'] = 2


# #### 4.6 Vintage

# In[27]:


print("Maximum vintage in months: {}".format(train["Vintage"].max()))
print("Minimum vintage in months: {}".format(train["Vintage"].min()))


# In[28]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Vintage',shade= True)
facet.set(xlim=(0, train['Vintage'].max()))
facet.add_legend()
plt.show() 


# Observing cross over points in the vintage vs is_lead attribute

# In[29]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Vintage',shade= True)
facet.set(xlim=(0, train['Vintage'].max()))
facet.add_legend()
plt.xlim(0, 10)


# In[30]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Vintage',shade= True)
facet.set(xlim=(0, train['Vintage'].max()))
facet.add_legend()
plt.xlim(7, 65)


# In[31]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Vintage',shade= True)
facet.set(xlim=(0, train['Vintage'].max()))
facet.add_legend()
plt.xlim(60, 135)


# #### 4.6.1 Binning - Vintage
# Binning/Converting Numerical Vintage to Categorical Variable
# 
# Feature vector map
# 
# Under 8 months old: 0
# 
# More the 8 and less than 62 months: 1
# 
# Greater than 62: 2

# In[32]:


for dataset in train_test_data:
    dataset.loc[ dataset['Vintage'] <= 8, 'Vintage'] = 0
    dataset.loc[(dataset['Vintage'] > 8) & (dataset['Vintage'] <= 62), 'Vintage'] = 1
    dataset.loc[ dataset['Vintage'] > 62, 'Vintage'] = 2


# #### 4.7 Average Account Banlance

# In[33]:


print("Maximum average account balance: {}".format(train["Avg_Account_Balance"].max()))
print("Minimum average account balance: {}".format(train["Avg_Account_Balance"].min()))


# In[34]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Avg_Account_Balance',shade= True)
facet.set(xlim=(0, train['Avg_Account_Balance'].max()))
facet.add_legend()
plt.show()


# In[35]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Avg_Account_Balance',shade= True)
facet.set(xlim=(0, train['Avg_Account_Balance'].max()))
facet.add_legend()
plt.xlim(0, 1000000)


# In[36]:


facet = sns.FacetGrid(train, hue="Is_Lead",aspect=4)
facet.map(sns.kdeplot,'Avg_Account_Balance',shade= True)
facet.set(xlim=(0, train['Avg_Account_Balance'].max()))
facet.add_legend()
plt.xlim(1000000, 3000000)


# #### 4.8.1 Binning - Average Account Balance
# Binning/Converting Numerical Average Account Balance to Categorical Variable
# 
# Feature vector map
# 
# Under 1000000 : 0
# 
# More the 1000000 and less than 3000000 months: 1
# 
# Greater than 3000000: 2

# In[37]:


for dataset in train_test_data:
    dataset.loc[ dataset['Avg_Account_Balance'] <= 1000000, 'Avg_Account_Balance'] = 0
    dataset.loc[(dataset['Avg_Account_Balance'] > 1000000) & (dataset['Vintage'] <= 3000000), 'Avg_Account_Balance'] = 1
    dataset.loc[ dataset['Avg_Account_Balance'] > 3000000, 'Avg_Account_Balance'] = 2


# In[38]:


train.head(40)


# #### 4.9 Region Code

# In[39]:


train["Region_Code"].value_counts()


# In[40]:


region_code_mapping = {"RG268": 0, "RG283": 1, "RG254": 1,
                 "RG284": 1, "RG277": 2, "RG280": 2,
                 "RG269": 3, "RG270": 3, "RG261": 3,
                 "RG257": 3, "RG251": 3, "RG282": 3,     
                 "RG274": 3, "RG272": 3, "RG281": 3,
                 "RG273": 3, "RG252": 3, "RG279": 3,
                 "RG263": 3, "RG275": 3,    "RG260": 3,
                 "RG256": 3,    "RG264": 3,    "RG276": 3,    
                 "RG259": 3,    "RG250": 3,    "RG255": 3,    
                 "RG258": 3,    "RG253": 3,    "RG278": 3,    
                 "RG262": 3,    "RG266": 3,    "RG265": 3, 
                 "RG271": 3,    "RG267": 3}


# In[41]:


for dataset in train_test_data:
    dataset['Region'] = dataset['Region_Code'].map(region_code_mapping)
    
train.drop("Region_Code",axis=1,inplace=True)
test.drop("Region_Code",axis=1,inplace=True)


# #### 4.10 Credit Product

# In[42]:


Pclass1 = train[train['Avg_Account_Balance']==0]['Credit_Product'].value_counts()
Pclass2 = train[train['Avg_Account_Balance']==1]['Credit_Product'].value_counts()
Pclass3 = train[train['Avg_Account_Balance']==2]['Credit_Product'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['Under 1^e6','betn 1^e6 & 3^e6', '> 3^e6']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# Since more the 50% under diffent average account brackets do not have credit product, fill out missing credit procut with "No" 

# In[43]:


for dataset in train_test_data:
    dataset['Credit_Product'] = dataset['Credit_Product'].fillna('No')


# In[44]:


credit_product_mapping = {"No": 0, "Yes": 1}
for dataset in train_test_data:
    dataset['Credit_Product'] = dataset['Credit_Product'].map(credit_product_mapping)


# In[45]:


train.head(50)


# In[46]:


train_data = train.drop(['Is_Lead','ID'], axis=1)
target = train['Is_Lead']

train_data.shape, target.shape


# In[47]:


# train_data.info()
train.isnull().sum()


# #### 5. Modelling

# In[48]:


from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np


# In[49]:


#k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
#clf = SVC()
#scoring = 'accuracy'
#score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)


# In[50]:


#round(np.mean(score)*100,2)


# #### 6. Testing

# In[ ]:


clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("ID", axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:


print(prediction)


# In[ ]:




