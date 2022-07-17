#!/usr/bin/env python
# coding: utf-8

# # Libraries import 

# Import the libraries necessary for this lab. We can add some aliases to make the libraries easier to use in our code and set a default figure size for further plots.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# #  Data Cleaning

# Let's read the data and look at the first 5 rows using the head method. The number of the output rows from the dataset is determined by the head method parameter.

# In[13]:


df = pd.read_excel('Retailers data.xlsx')
df.head(5)


# ### Identify missing values 
We will use the following function to identify missing data

(1) .isnull()

(2) .notnull()

The output is a boolean value indicating whether the value that is passed into the argument is indeed missing data.
# In[15]:


missing_data = df.isnull()
missing_data.head(5)

"False" indicates that the value is not missing, however this might not be true for the entire dataset since we only printed the first 5 rows.
# #### Count the missing values in each column

# In[18]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    


# Based on the results above, each column has 2240 rows of data and only one of the columns contain missing data:
# 
# "Income" : 24 missing data

# ## Dealing with missing data 

# The are two methods that we can use to deal with missing data:
# 
# (1) Drop data
# 
# (a) Drop the whole row
# 
# (b) Drop the whole column
# 
# (2) Replace data 
# 
# (a) Replace it by mean 
# 
# (b) Replace it by frequency
# 
# (c) Replace it based on other functions
# 
# Whole columns should be droppped if most entries are empty. In our dataset, none of the columns are empty enough to be dropped entirely. We will apply the replace data with mean method to the "Income" column.

# In[19]:


# Calculate the mean for the Income column
avg_income = df["Income"].astype("float").mean()
print("Average Income:", avg_income)


# In[20]:


# Replace NaN values with the mean value in the "Income" column
df["Income"].replace(np.nan, avg_income, inplace=True)


# ## Correct data format

# The last step in cleaning our data is to ensure that all data is in the correct format (int, float, category or other)
# 
# In pandas, we use
# 
# .dtype() to check the data type

# The last step in cleaning our data is to ensure that all data is in the correct format (int, float, category or other)
# 
# In pandas, we use
# 
# .dtype() to check the data type
# 
# .astype() to change the data type

# Let's list the data type for each column

# In[25]:


df.dtypes


# Now we have obtained a cleaned dataset with no missing values with all the data in its proper format.

# In[ ]:





# # Dataset exploration

# In this section we will explore the sourse dataset.

# Let's read the data and look at the first 5 rows using the head method. The number of the output rows from the dataset is determined by the head method parameter.

# In[2]:


df = pd.read_excel('Retailers data.xlsx')
df.head(5)


# ### Let's look at the dataset size, feature names and their types

# In[3]:


df.shape


# The dataset contains 2240 objects(rows), for each of which 29 features are set(columns)

# ### Attributing information

# Output the column(feature) names:

# In[4]:


df.columns


# To see the general information on all the DataFrame features (columns), we use the info method:

# In[5]:


df.info()


# The dataset contains 25 integer (int64), 1 real (float64) and 3 categorical and binary (object) features.

# Method describe shows the main statistical characteristics of the dataset for each numerical feature (int64 and float64 types): the existing values number, mean, standard deviation, range, min & max, 0.25, 0.5 and 0.75 quartiles.

# In[7]:


df.describe()


# The Mean row shows the feature average, STD is an RMS (Root Mean Square) deviation, min,max - the minimum and maximum values, 25%, 50%, 75%- quarters that split the dataset (or part of it) into four groups containing approximately an equal number of observations (rows). 

# To see the statistics on non-numeric features, you need to explicitly specify the feature types by the include parameter. You can also set include = all to output statistics on all the existing features.

# In[8]:


df.describe(include='all')


# The result shows that the average client refer is married (Marital_Status = Married) and has a university degree (Education = Graduation).

# ### Sorting 

# A DataFrame can be sorted by a few feature values. In our case, for example, by Year_Birth (ascending = True for sorting in ascending order):

# In[11]:


df.sort_values(by = "Year_Birth", ascending = True).head()


# The sorting results show that the oldest calls customer was born in 1893 and is Single, in addition has an income of 60182.0 $\$$

# ## Pivot tables

# In Pandas, pivot tables are implemented by the method pivot_table with such parameters:
# 
#     values – a list of variables to calculate the necessary statistics,
#     index – a list of variables to group data,
#     aggfunc — values that we actually need to count by groups - the amount, average, maximum, minimum or something else.

# Let's find the average income for different types of customer education

# In[47]:


df.pivot_table(values=["Income"],index=["Education"],aggfunc = "mean").head(5)


# The results above show that the average income increases with an increasing level of education

# ## Visualization in Pandas

# In[48]:


pd.plotting.scatter_matrix(
    df[["Year_Birth", "Income"]],
    figsize = (15, 15),
    diagonal = "kde")
plt.show()


# A scatter matrix (pairs plot) compactly plots all the numeric variables we have in a dataset against each other. The plots on the main diagonal allow you to visually define the type of data distribution: the distribution is similar to normal for age, and for a call duration and the number of contacts, the geometric distribution is more suitable.

# #### We also build a separate histogram for each feature:

# In[60]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10, 10)
df["Income"].hist()
plt.title('Income distribution')
plt.xlabel('Income')


# The histogram shows that around 40 % of the customers earn between 0 - 55000 $\$$, while around 40 % - 60% of the customers earn between 55000 $\$$ - 120000 $\$$ with the remaining portion earning between 120000 $\$$ - 200000 $\$$.

# #### Or we can also build it for all together:

# In[66]:


df.hist(color = "k",
        bins = 10,
        figsize = (20, 20))
plt.show()


# A visual analysis of the histograms presented allows us to make preliminary assumptions about the variability of the source data.

# While plots such as histogram are used mostly for continous variables, we may also want to investigate the relationship between 
# categorical variables. One of the plots used to represent categorical variables is the Box and whisker plot.

# Box and whisker plot is useful too. It allows you to compactly visualize the main characteristics of the feature distribution (the median, lower and upper quartile, minimal and maximum, outliers).

# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10, 10)
df.boxplot(column = "Income",
           by = "Marital_Status")
plt.show()


# The plot shows that Single customers on average earn less than the ones who are Married, together or Divorced. Whereas the ones belonging to the 'absurd' group on average earn higher when compared to the other groups. There is an outlier zone for the group s('Absurd', 'Alone' and 'Widow') over 100000 $\$$, and for the groups ('Divorced', 'Married', 'Single' and 'Together') over 200000 $\$$. 

# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10, 10)
df.boxplot(column = "Income",
           by = "Education")
plt.show()


# The plot shows that customers with Basic education on average earn less than the customers belonging to the other groups. Whereas the ones belonging to the 'PhD' group on average earn higher when compared to the other groups. There is an outlier zone for the groups ('2n Cycle' and 'Basic') over 100000 $\$$ , and for the groups ('Graduation', 'Master'and 'PhD') over 200000 $\$$. 

# In[ ]:




