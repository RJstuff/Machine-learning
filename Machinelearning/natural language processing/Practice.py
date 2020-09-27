import pandas as pd
df = pd.read_json("Restaurent_Reviews",lines=True)
df.head()
print(df)
