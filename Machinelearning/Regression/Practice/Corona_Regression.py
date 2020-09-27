import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
import plotly.graph_objects as go
data = pd.read_csv("confirmed cases - world wide - who.csv")
data["Date"] = pd.to_datetime(data["Date"], format=None)
#print(data.head())
print(data.columns)
df = pd.melt(data, id_vars="Date", value_vars=['China', 'Hong Kong', 'Macau', 'Taipei', 'Japan', 'South Korea',
       'Viet Nam', 'Singapore', 'Australia', 'Malaysia', 'Cambodia',
       'Philippines', 'Thailand', 'Nepal', 'Sri Lanka', 'India', 'USA',
       'Canada', 'France', 'Finland', 'Germany', 'Italy', 'Russia', 'Spain',
       'Sweden', 'UK', 'Belgium', 'UAE'],var_name="Country", value_name="Death_count")
#print(df.head())
#print(df.info())
data_t = data.T
data_t.columns = data_t.iloc[0]
print(data_t.columns)
data_t.drop(data_t.index[0], inplace=True)
data_t.sort_values('2/5/2020', ascending=False, inplace=True)
data_t = data_t.astype('int32')
data_t.style.background_gradient(cmap='viridis')
print(data_t)

w_china = df[df['Date']==max(df['Date'])].sort_values("Death_count", ascending=False).reset_index(drop=True)[['Country','Death_count']].iloc[1:]
print(w_china)

