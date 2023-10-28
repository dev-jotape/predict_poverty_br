import pandas as pd

df = pd.read_csv('./excel-files/cities_indicators.csv')
group = df.groupby('city_code').first()
print(group)
print(group['income'].mean())
