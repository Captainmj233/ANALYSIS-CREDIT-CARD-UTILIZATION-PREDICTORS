#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
import statsmodels.api as sm


# In[2]:


#Load the credit card dataset
credit_card_data= pd.read_csv('/Users/captainmj/Downloads/BankChurners.csv')


# In[3]:


credit_card_data.head()


# In[4]:


credit_card_data.shape


# In[5]:


credit_card_data.columns


# In[6]:


credit_card_data = credit_card_data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)


# In[7]:


credit_card_data.describe().T


# In[8]:


credit_card_data.hist(figsize=(15,15))
plt.show()


# In[13]:


# Calculate the mean credit limit for each card category
card_category_means = credit_card_data.groupby('Card_Category')['Credit_Limit'].mean()

# Plot the bar plot
plt.bar(card_category_means.index, card_category_means.values)
plt.xlabel('Card Category')
plt.ylabel('Average Credit Limit')
plt.title('Average Credit Limit by Card Category')

# Add text annotations to the bars
for i, val in enumerate(card_category_means.values):
    plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')


# Show the plot
plt.show()


# In[34]:




# Calculate the count of each card category
card_category_counts = credit_card_data['Card_Category'].value_counts()

# Create a list of colors for each category
colors = ['blue', 'green', 'orange', 'red']  # Add more colors if needed

# Create a bar chart with different colors for each category
plt.bar(card_category_counts.index, card_category_counts.values, color=colors)
plt.xlabel('Card Category')
plt.ylabel('Card_Ca')
plt.title('Total Count of Each Card Category')

# Add count values as text annotations on top of each bar
for i, count in enumerate(card_category_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()


# In[35]:


# Calculate the count of customers by income category
income_counts = credit_card_data['Income_Category'].value_counts()

# Create a pie chart of the count of customers by income category
plt.pie(income_counts, labels=income_counts.index, autopct='%1.1f%%')
plt.title('Count of Customers by Income Category')
plt.show()


# In[13]:



#Produce a correlation matrix of the numeric values.
corr_matrix =credit_card_data.corr()


# In[14]:


#plot of the correlation matrix
plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Randomly sample 500 data points
sample_data = credit_card_data.sample(500)

# Regression plot for 'Customer_Age' vs 'Avg_Utilization_Ratio'
sns.regplot(x='Customer_Age', y='Avg_Utilization_Ratio', data=sample_data, line_kws={"color": "red"})
plt.show()

# Regression plot for 'Credit_Limit' vs 'Avg_Utilization_Ratio'
sns.regplot(x='Credit_Limit', y='Avg_Utilization_Ratio', data=sample_data, line_kws={"color": "red"})
plt.show()

# Regression plot for 'Revolving_Balance' vs 'Avg_Utilization_Ratio'
sns.regplot(x='Total_Revolving_Bal', y='Avg_Utilization_Ratio', data=sample_data, line_kws={"color": "red"})
plt.show()


# In[17]:


import statsmodels.api as sm

# Select the independent variables (customer age, credit limit, revolving balance) 
# and the dependent variable (credit card utilization ratio)
X = credit_card_data[['Customer_Age', 'Credit_Limit', 'Total_Revolving_Bal']]
Y = credit_card_data['Avg_Utilization_Ratio']

# Adding constant to the independent variables
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())


# In[25]:


## Customer_Age
## Null Hypothesis (H0): There is no significant relationship between Customer_Age and Avg_Utilization_Ratio.
## Alternative Hypothesis (H1): There is a significant relationship between Customer_Age and Avg_Utilization_Ratio..

## Credit_Limit
## Null Hypothesis (H0): There is no significant relationship between Credit_Limit and Avg_Utilization_Ratio.
## Alternative Hypothesis (H1): There is a significant relationship between Credit_Limit and Avg_Utilization_Ratio.

## Total_Revolving_Bal
## Null Hypothesis (H0): There is no significant relationship between Total_Revolving_Bal and Avg_Utilization_Ratio.
## Alternative Hypothesis (H1): There is a significant relationship between Total_Revolving_Bal and Avg_Utilization_Ratio.


# In[27]:



# fetch p-values for each coefficient
p_values = model.pvalues

# print results of hypothesis tests
for variable, p_value in p_values.iteritems():
    print(f"Results for {variable}:")
    if p_value < alpha:
        print(f"The p-value {p_value} is less than the significance level {alpha}, thus we reject the null hypothesis. There is a significant relationship between {variable} and 'Avg_Utilization_Ratio'.")
    else:
        print(f"The p-value {p_value} is greater than the significance level {alpha}, thus we fail to reject the null hypothesis. There may not be a significant relationship between {variable} and 'Avg_Utilization_Ratio'.")
    print()


# In[ ]:




