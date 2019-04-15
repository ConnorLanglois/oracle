import pandas as pd

from os import path
from numpy import random
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

random.seed(0)

df = pd.read_csv(path.relpath('data/aapl.csv'))[::-1]
features = df.columns.difference(['Date', 'Correction'])

x = StandardScaler().fit_transform(df[features].astype(float))
y = df['Correction'].values

series = TimeseriesGenerator(x, y, 14, batch_size=len(x))[0]
x_series = series[0]
y_series = series[1]

train_p = 0.90
index = int(train_p * len(df))

train_x = x_series[:index]
train_y = y_series[:index]
test_x = x_series[index:]
test_y = y_series[index:]

model = Sequential()
model.add(LSTM(10))
