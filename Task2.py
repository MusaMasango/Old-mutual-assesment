#!/usr/bin/env python
# coding: utf-8

# # Libraries import 

# Import the libraries necessary for this lab. We can add some aliases to make the libraries easier to use in our code and set a default figure size for further plots.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


# #  Data Cleaning

# Let's read the data and look at the first 5 rows using the head method. The number of the output rows from the dataset is determined by the head method parameter.

# In[2]:


df = pd.read_csv('bank data.csv')
df.head(5)


# ### Identify missing values 

# We will use the following function to identify missing data
# 
# (1) .isnull()
# 
# (2) .notnull()
# 
# The output is a boolean value indicating whether the value that is passed into the argument is indeed missing data.

# In[3]:


missing_data = df.isnull()
missing_data.head(5)


# "False" indicates that the value is not missing, however this might not be true for the entire dataset since we only printed the first 5 rows.

# #### Count the missing values in each column

# In[4]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    


# Based on the results above, each column has 11162 rows of data and only one of the columns contain no missing data

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

# In[5]:


# Calculate the mean for the Income column
#avg_income = df["Income"].astype("float").mean()
#print("Average Income:", avg_income)


# In[6]:


# Replace NaN values with the mean value in the "Income" column
#df["Income"].replace(np.nan, avg_income, inplace=True)


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

# In[7]:


df.dtypes


# Now we have obtained a cleaned dataset with no missing values with all the data in its proper format.

# # Dataset exploration

# In this section we will explore the sourse dataset.

# Let's read the data and look at the first 5 rows using the head method. The number of the output rows from the dataset is determined by the head method parameter.

# In[8]:


df = pd.read_csv('bank data.csv')
df.head(5)


# ### Let's look at the dataset size, feature names and their types

# In[9]:


df.shape


# The dataset contains 2240 objects(rows), for each of which 17 features are set(columns)

# ### Attributing information

# Output the column(feature) names:

# In[10]:


df.columns


# To see the general information on all the DataFrame features (columns), we use the info method:

# In[11]:


df.info()

The dataset contains 7 integer (int64) and 10 categorical and binary (object) features.
# The method describe shows the main statistical characteristics of the dataset for each numerical feature (int64 and float64 types): the existing values number, mean, standard deviation, range, min & max, 0.25, 0.5 and 0.75 quartiles.

# In[12]:


df.describe()


# The Mean row shows the feature average, STD is an RMS (Root Mean Square) deviation, min,max - the minimum and maximum values, 25%, 50%, 75%- quarters that split the dataset (or part of it) into four groups containing approximately an equal number of observations (rows). 

# To see the statistics on non-numeric features, you need to explicitly specify the feature types by the include parameter. You can also set include = all to output statistics on all the existing features.

# In[13]:


df.describe(include='all')


# The result shows that the average client refers to administrative staff (job = management.), is married (marital = married) and has a university degree (education = secondary).

# For categorical (type object) and boolean (type bool) features you can use the value_counts method. Let's look at the target feature (deposit) distribution:

# In[14]:


df["deposit"].value_counts()


# 5289 clients (47.4%) of 11162 issued a term deposit, the value of the variable 'deposit' equals 'yes'.
# 
# Let's look at the client distribution by the variable `marital`. Specify the value of the `normalize = True` parameter to view relative frequencies, but not absolute.

# In[15]:


df["marital"].value_counts(normalize = True)


# As we can see, 57% (0.57) of clients are married, which must be taken into account when planning marketing campaigns to manage deposit operations.

# In[16]:


df["education"].value_counts()


# ### Sorting 

# A DataFrame can be sorted by a few feature values. In our case, for example, by duration (ascending = True for sorting in ascending order):

# In[17]:


df.sort_values(by = "duration", ascending = False).head()


# The sorting results show that the longest calls exceed one hour, as the value duration is more than 3600 seconds or 1 hour. At the same time, it usually was on day 9 and, especially, in June (month).

# ## Application of functions: apply, map etc.

# #### Apply the max function to each column:

# In[18]:


df.apply(np.max)


# The oldest client is 95 years old (age = 95), and the number of contacts with one of the customers reached 63 (campaign = 63).

# The apply method can also be used to apply the function to each row. To do this, you need to specify the axis = 1.

# #### Apply the map function to each column:

# The map can also be used for the values replacement in a column by passing it as an argument dictionary in form of {old_value: new_value}.

# In[19]:


#mapping for the deposit column
a= {"no": 0, "yes": 1}
df["deposit"] = df["deposit"].map(a)
df.head()


# ## Pivot tables
Suppose we want to see how observations in our sample are distributed in the context of two features - deposit and marital. To do this, we can build cross tabulation by the crosstab method.
# In[20]:


pd.crosstab(df["deposit"], df["marital"], normalize='index')

We see that more than half of the clients (52%, column married) are married and have issued a deposit.
# In Pandas, pivot tables are implemented by the method pivot_table with such parameters:
# 
#     values – a list of variables to calculate the necessary statistics,
#     index – a list of variables to group data,
#     aggfunc — values that we actually need to count by groups - the amount, average, maximum, minimum or something else.

# Let's find the average age and the call duration for different types of client employment job:

# In[21]:


df.pivot_table(values=["age","duration"],index=["job"],aggfunc = "mean").head(5)


# The obtained results allow you to plan marketing banking campaigns more effectively.

# ## Visualization in Pandas

# In[22]:


pd.plotting.scatter_matrix(
    df[["age", "duration","campaign"]],
    figsize = (15, 15),
    diagonal = "kde")
plt.show()


# A scatter matrix (pairs plot) compactly plots all the numeric variables we have in a dataset against each other. The plots on the main diagonal allow you to visually define the type of data distribution: the distribution is similar to normal for age, and for a call duration and the number of contacts, the geometric distribution is more suitable.

# #### We also build a separate histogram for each feature:

# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10, 10)
df["age"].hist()
plt.title('Income distribution')
plt.xlabel('age')

The histogram shows that most of our clients are between the ages of 25 and 50, which corresponds to the actively working part of the population.
# #### Or we can also build it for all together:

# In[24]:


df.hist(color = "k",
        bins = 10,
        figsize = (20, 20))
plt.show()


# A visual analysis of the histograms presented allows us to make preliminary assumptions about the variability of the source data.

# While plots such as histogram are used mostly for continous variables, we may also want to investigate the relationship between 
# categorical variables. One of the plots used to represent categorical variables is the Box and whisker plot.

# Box and whisker plot is useful too. It allows you to compactly visualize the main characteristics of the feature distribution (the median, lower and upper quartile, minimal and maximum, outliers).

# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10, 10)
df.boxplot(column = "age",
           by = "marital")
plt.show()


# The plot shows that unmarried people are on average younger than divorced and married ones. For the last group, there is an outlier zone over 70 years old.

# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10, 10)
df.boxplot(column = "age",
           by = ["marital","housing"])
plt.show()


# As you can see, age and marital status do not have any significant influence on having a housing loan

# In[27]:


df = df.replace({"admin.": 1, "blue-collar": 2, "entrepreneur": 3, "housemaid": 4, "management": 5, "retired": 6, "self-employed": 7, "services": 8,"student": 9, "technician": 10, "unemployed": 11, "unknown": 12})
df


# In[28]:


df = df.replace({"divorced": 1, "married": 2, "single": 3, "unknown":4 })
df


# In[29]:


df = df.replace({"primary": 1, "secondary": 2, "tertiary": 3, "unknown": 4})
df


# # Model Development

# In this section we will develop a model that will predict which customers are likely to take up the fixed deposit.
# 
# Some questions we want to answer for this problem include
# 
# (a) Which features have the greatest impact?
# (b) how do they impact the prediction?
# 
# 

# A model will help us understand the exact relationship between different variables and how these variables are used to predict the result.

# ### Importing libraries

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing


# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:

# In[40]:


X = df[['age', 'job','marital', 'education','duration','campaign']] .values
X[0:5]


# In[41]:


y=df['deposit'].values

Y[0:5]


# ## Normalize data

# Data Standardization gives the data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on the distance of data points:

# In[42]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# ### Train Test Split
# 
# Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that the model has NOT been trained on. Doing a train and test on the same dataset will most likely have low out-of-sample accuracy, due to the likelihood of our model overfitting.
# 
# It is important that our models have a high, out-of-sample accuracy, because the purpose of any model, of course, is to make correct predictions on unknown data. So how can we improve out-of-sample accuracy? One way is to use an evaluation approach called Train/Test Split. Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set.
# 
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that has been used to train the model. It is more realistic for the real world problems.
# 

# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# ## Classification Algorithm(K nearest neighbor)
# 
# 

# In[ ]:


Since we have categorical variables in our data, it is better for us to rather use a classification algorithm when building our model. On of the algorithms that we will use is the K nearest neighbor algorithm.


# ### K nearest neighbor (KNN)

# K-Nearest Neighbors is a supervised learning algorithm. Where the data is 'trained' with data points corresponding to their classification. To predict the class of a given data point, it takes into account the classes of the 'K' nearest data points and chooses the class in which the majority of the 'K' nearest data points belong to as the predicted class.

# In this sense, it is important to consider the value of k. It considers the 'K' Nearest Neighbors (data points) when it predicts the classification of the test point.

# ##### Import library

# Classifier implementing the k-nearest neighbors vote.

# In[44]:


from sklearn.neighbors import KNeighborsClassifier


# In[45]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[46]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[47]:


#Model evaluation
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[48]:


k = 6
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
yhat6 = neigh6.predict(X_test)
# model evaluation
print('Train set accuracy: ',metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print('Test set accuracy: ', metrics.accuracy_score(y_test, yhat6))


# In[49]:


Ks = 10
mean_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

mean_acc


# ## Model Evaluation Using Visualization

# In[50]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.title('Accuracy plot')
plt.tight_layout()
plt.show()


# In[51]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# It can be seen from the above plot that the features that have the greatest impact on the deposit (with a certain degree of accuracy are):
#     
#     age
#     
#     job
#     
#     marital
#     
#     education
#     
#     duration
#     
#     campaign

# Some of the challenges that I encoutered when completing this task include:
# (1) Since most of the data entries were categorical variables, I first had to convert them into numerical values so that I can be able to build a model.

# In[ ]:




