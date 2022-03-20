import pandas as pd
data=pd.read_csv('census.csv', skipinitialspace=True)  
print(data.head())
data.to_csv("preprocessed_data.csv")