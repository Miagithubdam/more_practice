import numpy as np 
import pandas as pd

dataset = pd.read_csv('C:\ITBAN4 SIR NEIL\data prepros')
print(dataset.describe())

x = dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

from sklearn.impute import simpl