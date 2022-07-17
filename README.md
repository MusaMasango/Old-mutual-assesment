For this assesment I am required to complete an data science assesment given by Old Mutual.  The assesment consits of 2 tasks, namely Task1 and Task2. The given tasks are outlined below.

Task 1
I was provided with a retailer’s data (task 1 folder).

The deliverables as required by the client:
•	Cluster the customers into clear segments.
o	The person normally responsible for keeping the data in order is off sick. Please point out any issues that emerge during data cleaning so that the client may remediate.
o	The client is new to data science, if you create any new features please highlight these so that the client may incorporate going forward.
o	The final delivery should include a powerpoint slide that clearly describes the segments eg. parents vs non, income groups etc.

I completed the task using jupyter notebook (with python being the programming language). The following processes where executed as part of the task :
(1) Import the required libraries (numpy, pandas, and matplotlib).
(2) read the data using the pd.read_excel() command.
(3) data cleaning: this includes checking for missing values with the data as well as ensuring that the data is in the correct format.
(4) data exploration: this includes the basic information of the data, basic statistical analysis and pivot tables.
(5) data visualization: this includes various plots such as scattering matrix, histogram and box and whisker plots.

Some of the challenges that I encoutered when completing this task include:
(1) Firstly for the given data I was given the birth year and not the age of the client, which made conducting the data analysis much more complicated.
(2) The Retailers data file could not be displayed properly in the notebook when it was in the csv format, hence I had to convert the file into an excel file.



Task 2
I was provided with a bank’s data. (task 2 folder).
The  deliverables as required by the client:
•	Build a model to predict which customers are likely to take up the fixed deposit.
•	Include an evaluation of the model performance. 
•	Determining which features have the greatest impact and how do they impact the prediction.

Similarly as I did with task 1, I completed the task using jupyter notebook (with python being the programming language). The following processes where executed as part of the task :
(1) Import the required libraries (numpy, pandas, matplotlib, and sklearn).
(2) read the data using the pd.read_csv() command.
(3) data cleaning: this includes checking for missing values with the data as well as ensuring that the data is in the correct format.
(4) data exploration: this includes the basic information of the data, basic statistical analysis and pivot tables.
(5) data visualization: this includes various plots such as scattering matrix, histogram and box and whisker plots.

Some of the challenges that I encoutered when completing this task include:
(1) Since most of the data entries were categorical variables, I first had to convert them into numerical values so that I can be able to build a model.
