 ## Old Mutual Assesment

**Objective**

For this assesment I am required to complete an data science assesment given by Old Mutual.  The assesment consits of 2 tasks, namely Task1 and Task2. The given tasks are outlined below.

**Task 1**

For this task, I was provided with a retailer’s data (task 1 folder).  The deliverables required by the client are as follows:

 - Cluster the customers into clear segments(i.e by education, marital status etc)
 - Highlighting out any issues that emerge during data cleaning.
 
 **Method**
 
 I completed the task using jupyter notebook (with python being the programming language). The following processes where executed as part of the task :
 - Import the required libraries (numpy, pandas, and matplotlib) as shown by the code below

    ```python 
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    ```
 - Read the data using the pd.read_excel() command as shown by the code below
 ```python
 df = pd.read_excel('Retailers data.xlsx')
df.head(5)
```
 - Data cleaning: this includes checking for missing values with the data as well as ensuring that the data is in the correct format.
 - Data exploration: this includes the basic information of the data, basic statistical analysis and pivot tables.
 - data visualization: this includes various plots such as scattering matrix, histogram and box and whisker plots.
 
 **Findings**
 
 - The results obtained from the pivot table show that the average income increases with an increasing level of education
 - The histogram plot shows that around 40 % of the customers earn between 0 - 55000 $\, while around 40 % - 60% of the customers earn between 55000 $\- 120000 $ with the remaining portion earning between 120000 $ - 200000 $\.
 - The Box and whisker plot shows that Single customers on average earn less than the ones who are Married, together or Divorced. Whereas the ones belonging to the 'absurd' group on average earn higher when compared to the other groups. There is an outlier zone for the groups(Absurd, Alone and Widow) over 100000 $\, and for the groups (Divorced, Married, Single and Together) over 200000 $\.

**Challenges**

 - Firstly for the given data I was given the birth year and not the age of the client, which made conducting the data analysis much more complicated.
 - The Retailers data file could not be displayed properly in the notebook when it was in the csv format, hence I had to convert the file into an excel file.
 
## Task 2
For this task, I was provided with a bank’s data. (task 2 folder). The  deliverables as required by the client are as follows:
 - Build a model to predict which customers are likely to take up the fixed deposit.
 - Include an evaluation of the model performance.
 - Determining which features have the greatest impact and how do they impact the prediction.

**Method**
Similarly as I did with task 1, I completed the task using jupyter notebook (with python being the programming language). The following processes where executed as part of the task :

 - Import the required libraries (numpy, pandas, matplotlib, and sklearn) as shown in the code below
```python
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn import preprocessing
```
 - Read the data using the pd.read_csv() command as shown in the code below
 ```python
 df = pd.read_csv('bank data.csv')
df.head(5)
```
 - Data cleaning: this includes checking for missing values with the data as well as ensuring that the data is in the correct format.
 - Data exploration: this includes the basic information of the data, basic statistical analysis and pivot tables.
 - Data visualization: this includes various plots such as scattering matrix, histogram and box and whisker plots.
 - Model development: this includes importing the required libraries(i.e sklearn) and splitting the data into a training and testing datasets then predicting the outcome(deposit) using the test dataset as shown by the code below
  ```python
  from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape, y_train.shape)

print ('Test set:', X_test.shape, y_test.shape)
```
 - Model evaluation: this includes metrics to evaluate the performance of our model such as accuracy as shown in the image below
```python
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
```
  
 **Findings**
 - The histogram shows that most of our clients are between the ages of 25 and 50, which corresponds to the actively working part of the population.
 - The Box and whisker plot shows that unmarried people are on average younger than divorced and married ones. For the last group, there is an outlier zone over 70 years old.
 - It can be seen from the accuracy plot that the features that have the greatest impact on the deposit (with a certain degree of accuracy are):
 - age
 - job
 - marital(Marital status)
 - education
 - duration
 - campaign

**Challenge**
 The challenge that I encoutered when completing this task is that Since most of the data entries were categorical variables, I first had to convert them into numerical values so that I can be able to build a model.

 

 

 

 


