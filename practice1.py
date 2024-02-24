
import pandas as pd
import numpy as np
import lib



#load dataset
dataset = pd.read_csv('activity1.csv')
print(dataset.describe())

#create dependent and independent variable
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)
#hundle missing data
#count the number of missing values in each column
print(dataset.isnull().sum())

#drop missing value record
dataset.dropna(inplace=True)