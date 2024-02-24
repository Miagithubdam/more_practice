#Activity 2
#1.) creating independent x and dependent y variables
#2.) Filling out missing data using mean x
#3.) filling our missing data using most_frequent in y
#4.) Converting categorical to nominal in x column 0
#5.) Converting categorical to nominal in y
#6.) splitting data into training and test
#7.) feature scaling - Standardization & Normalization for x
#8.) feature scaling - Standardization & Normalization for y


# Importing libraries
import numpy as np
 import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
dataset = pd.read_csv('dataset2.csv')
print(dataset.describe())

#1.) creating independent x and dependent y variables
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)



#2.) Filling out missing data using mean x
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])



#3.) filling our missing data using most_frequent in y
imputer = SimpleImputer(strategy='most_frequent')
dataset.iloc[:, 3] = imputer.fit_transform(dataset.iloc[:, 3].values.reshape(-1, 1)).ravel()


# viewing the graph of age data
plt.hist(dataset['Age'],bins=15) #bins = number of bar graph / intervals
plt.show()


#quantile

# Lower percentiles (e.g., 1st or 5th percentile): These are useful for identifying outliers 
# at the lower end of the distribution. They are often employed when data is expected to have 
# a lower bound or when you want to identify extreme low values.
age_lowerLimit = dataset['Age'].quantile(0.05)
age_lowerLimit

# Upper percentiles (e.g., 95th or 99th percentile): These are useful for identifying outliers 
# at the upper end of the distribution. They are often used when you want to detect extreme 
# high values or when the data may have an upper limit
age_upperLimit = dataset['Age'].quantile(0.95)
age_upperLimit



# Scatter plot of age column
plt.scatter(dataset.index, dataset['Age'], label='Age')
# Plot outliers
outliers = dataset[(dataset['Age'] > age_upperLimit ) | (dataset['Age'] < age_lowerLimit)]
plt.scatter(outliers.index, outliers['Age'], color='red', label='Outliers')
# Draw lines for upper and lower limits
for index, row in outliers.iterrows():
    plt.axhline(y=row['Age'], color='gray', linestyle='--')

plt.axhline(y=age_lowerLimit, color='orange', linestyle='--', label='Lower Limit')
plt.axhline(y=age_upperLimit, color='yellow', linestyle='--', label='Upper Limit')
plt.xlabel('Index')
plt.ylabel('Age')
plt.title('Scatter Plot of Age with Outliers')
plt.legend()
plt.grid(True)
plt.show()