import pandas as pd

from os import path
from numpy import random
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

random.seed(0)

df = pd.read_csv(path.relpath('data/aapl.csv'))[::-1]
features = df.columns.difference(['Date', 'Correction'])

x = StandardScaler().fit_transform(df[features].astype(float))
y = df['Correction'].values

series = TimeseriesGenerator(x, y, 14, batch_size=len(x))[0]
series_x = series[0]
series_y = series[1]

train_p = 0.90
index = int(train_p * len(df))

train_x = series_x[:index]
train_y = series_y[:index]
test_x = series_x[index:]
test_y = series_y[index:]

model = Sequential()
model.add(LSTM(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100)

metrics = model.evaluate(test_x, test_y)
predictions = model.predict(test_x)

print('\nAccuracy: ', metrics[1])
print('\nPredictions:\n', predictions)
