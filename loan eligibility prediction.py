#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import seaborn as sns
sns.set_style('dark')


# In[5]:


loan_train = pd.read_csv("E:\loan dataset\loan-train.csv")


# In[6]:


loan_train.head()


# In[7]:


loan_test=pd.read_csv("E:\loan dataset\loan-test.csv")


# In[8]:


loan_test.head()


# In[9]:


print("Rows: ", len(loan_train))


# In[10]:


print("Columns: ", len(loan_train.columns))


# In[11]:


print("Shape : ", loan_train.shape)


# In[12]:


print(loan_train.columns)


# In[13]:


loan_train.describe()


# In[14]:


loan_train.info()


# In[15]:


# Exploratory Data Analysis
def explore_object_type(df ,feature_name):
    if df[feature_name].dtype ==  'object':
        print(df[feature_name].value_counts())


# In[ ]:





# In[ ]:





# In[16]:


# Now, Test and Call a function for gender only
explore_object_type(loan_train, 'Gender')


# In[17]:


for featureName in loan_train.columns:
    if loan_train[featureName].dtype == 'object':
        print('\n"' + str(featureName) + '\'s" Values with count are :')
        explore_object_type(loan_train, str(featureName))


# In[18]:


import missingno as msno


# In[19]:


# list of how many percentage values are missing
loan_train

loan_train.isna().sum()
# round((loan_train.isna().sum() / len(loan_train)) * 100, 2)


# In[20]:


msno.bar(loan_train)


# In[21]:


msno.matrix(loan_train )


# In[22]:


loan_train['Credit_History'].fillna(loan_train['Credit_History'].mode(), inplace=True) # Mode
loan_test['Credit_History'].fillna(loan_test['Credit_History'].mode(), inplace=True) # Mode


loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].mean(), inplace=True) # Mean
loan_test['LoanAmount'].fillna(loan_test['LoanAmount'].mean(), inplace=True) # Mean


# In[23]:


# convert Categorical variable with Numerical values.


# In[24]:


loan_train.Loan_Status = loan_train.Loan_Status.replace({"Y": 1, "N" : 0})


# In[25]:


loan_train.Gender = loan_train.Gender.replace({"Male": 1, "Female" : 0})
loan_test.Gender = loan_test.Gender.replace({"Male": 1, "Female" : 0})

loan_train.Married = loan_train.Married.replace({"Yes": 1, "No" : 0})
loan_test.Married = loan_test.Married.replace({"Yes": 1, "No" : 0})

loan_train.Self_Employed = loan_train.Self_Employed.replace({"Yes": 1, "No" : 0})
loan_test.Self_Employed = loan_test.Self_Employed.replace({"Yes": 1, "No" : 0})


# In[26]:


loan_train['Gender'].fillna(loan_train['Gender'].mode()[0], inplace=True)
loan_test['Gender'].fillna(loan_test['Gender'].mode()[0], inplace=True)

loan_train['Dependents'].fillna(loan_train['Dependents'].mode()[0], inplace=True)
loan_test['Dependents'].fillna(loan_test['Dependents'].mode()[0], inplace=True)

loan_train['Married'].fillna(loan_train['Married'].mode()[0], inplace=True)
loan_test['Married'].fillna(loan_test['Married'].mode()[0], inplace=True)

loan_train['Credit_History'].fillna(loan_train['Credit_History'].mean(), inplace=True)
loan_test['Credit_History'].fillna(loan_test['Credit_History'].mean(), inplace=True)


# In[29]:


from sklearn.preprocessing import LabelEncoder
feature_col = ['Property_Area','Education', 'Dependents']
le = LabelEncoder()
for col in feature_col:
    loan_train[col] = le.fit_transform(loan_train[col])
    loan_test[col] = le.fit_transform(loan_test[col])


# In[ ]:


#Data Visualizations


# In[31]:


loan_train


# In[32]:


loan_train.plot(figsize=(18, 8))

plt.show()


# In[33]:


plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)


loan_train['ApplicantIncome'].hist(bins=10)
plt.title("Loan Application Amount ")

plt.subplot(1, 2, 2)
plt.grid()
plt.hist(np.log(loan_train['LoanAmount']))
plt.title("Log Loan Application Amount ")

plt.show()


# In[34]:


plt.figure(figsize=(18, 6))
plt.title("Relation Between Applicatoin Income vs Loan Amount ")

plt.grid()
plt.scatter(loan_train['ApplicantIncome'] , loan_train['LoanAmount'], c='k', marker='x')
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()


# In[35]:


plt.figure(figsize=(12, 6))
plt.plot(loan_train['Loan_Status'], loan_train['LoanAmount'])
plt.title("Loan Application Amount ")
plt.show()


# In[36]:


plt.figure(figsize=(12,8))
sns.heatmap(loan_train.corr(), cmap='coolwarm', annot=True, fmt='.1f', linewidths=.1)
plt.show()


# In[ ]:


# Choose ML Model


# In[38]:


# import ml model from sklearn pacakge

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
logistic_model = LogisticRegression()


# In[ ]:


#Traing the ML Model


# In[39]:


train_features = ['Credit_History', 'Education', 'Gender']

x_train = loan_train[train_features].values
y_train = loan_train['Loan_Status'].values

x_test = loan_test[train_features].values


# In[40]:


logistic_model.fit(x_train, y_train)


# In[ ]:


# Predict Model


# In[41]:


# Predict the model for testin data

predicted = logistic_model.predict(x_test)


# In[42]:


# check the coefficeints of the trained model
print('Coefficient of model :', logistic_model.coef_)


# In[43]:


# check the intercept of the model
print('Intercept of model',logistic_model.intercept_)


# In[44]:


# Accuray Score on train dataset
# accuracy_train = accuracy_score(x_test, predicted)
score = logistic_model.score(x_train, y_train)
print('accuracy_score overall :', score)
print('accuracy_score percent :', round(score*100,2))


# In[45]:


# predict the target on the test dataset
predict_test = logistic_model.predict(x_test)
print('Target on test data',predict_test) 


# In[ ]:




