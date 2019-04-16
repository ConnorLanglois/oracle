from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

np.random.seed(0)

df = pd.read_csv(Path() / 'data/aapl.csv')[::-1]

x = StandardScaler().fit_transform(df[df.columns.difference(['Date', 'Correction'])].astype(float))
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
