import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras import regularizers
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, CuDNNLSTM
from keras.preprocessing.sequence import TimeseriesGenerator

sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
np.random.seed(0)

df = pd.read_csv('../data/aapl.csv')[::-1]

x = StandardScaler().fit_transform(df[df.columns.difference(['Date', 'Correction'])].astype(float))
y = df['Correction'].values

batch_size = 353
timesteps = 14
test_n = 353

series = TimeseriesGenerator(x, y, timesteps, batch_size=len(x))[0]
series_x = series[0]
series_y = series[1]

train_x = series_x[:-test_n]
train_y = series_y[:-test_n]
test_x = series_x[-test_n:]
test_y = series_y[-test_n:]

model = Sequential()
model.add(CuDNNLSTM(10, batch_input_shape=(batch_size, series_x.shape[1], series_x.shape[2]), stateful=True))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for i in range(1, 1001):
	model.fit(train_x, train_y, batch_size=batch_size, epochs=1, shuffle=False, verbose=0)

	train_acc = model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)[1]
	test_acc = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)[1]

	print(model.predict(test_x, batch_size=batch_size, verbose=0))

	print(i, round(train_acc, 2), round(test_acc, 2))

	model.reset_states()
