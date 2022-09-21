import pandas as pd

data = pd.read_csv('data\PoetryFoundationData.csv')
data = data.dropna()
data = data['Poem'].str.lower()

string = ''
for x in data:
    string += x + "</s>"

with open('data\poetry.txt', 'w', encoding='utf-8') as f:
    f.write(string)
