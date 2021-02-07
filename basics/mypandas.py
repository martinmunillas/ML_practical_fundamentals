import pandas as pd

series = pd.Series([5, 10, 15, 20, 25])
print(series)

print(series[3])

df = pd.DataFrame(['Hello', 'world', 'Robotic'])
print(df)

data = pd.DataFrame({ 'name': ['Jhon', 'Anne', 'Joseph', 'Arthur'], 'age': [14, 23, 56, 23], 'country': ['Chile', 'Argentina', 'Chile', 'Mexico']})
print(data)
print()

print(data[['name', 'country']])
print()

actors = pd.read_csv('asdf.csv')
print(actors.head(5))
print()

movie = actors.movie
print(movie)
print()

lizzi = actors.iloc[0]['name']
print(lizzi)

print(actors.shape)
print()
print(actors.columns)
print()
print(actors['name'].describe())
print()
print(actors.sort_index(axis=0, ascending=False))
print()