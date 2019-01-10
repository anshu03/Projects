#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Importing the dataset
data = pd.read_csv('MarketBasketOptimisation.csv',header=None)
transactions = []
for i in range(7501):
    transactions.append([str(data.values[i,j]) for j in range(20)])

#Training the Apriori on the dataset
from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#Visualising the results
rules = list(rules)
[print(x) for x in rules]