import pandas as pd

df = pd.read_csv("data/Tweets.csv")

print(df.head())
print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns)