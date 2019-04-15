import pandas as pd

from os import path
from numpy import random
from keras.models import Sequential
from keras.layers import LSTM

random.seed(0)

df = pd.read_csv(path.relpath('data/aapl.csv'))[::-1]
x = df.loc[:, df.columns != 'Correction'].values
y = df['Correction'].values

print(df[df.columns != 'Correction'])
print(x, y)

train_p = 0.90
index = int(train_p * len(df))
train = df[:index]
test = df[index:]

model = Sequential()
model.add(LSTM(10))
