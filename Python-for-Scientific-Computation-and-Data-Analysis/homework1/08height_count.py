import numpy as np
import pandas as pd

input_df = pd.read_csv('height.csv')

input_df['age'] = input_df['age'].astype(int)

count = np.bincount(input_df['age'])
h_sum = np.bincount(input_df['age'], weights=input_df['height'])

result = h_sum/count
