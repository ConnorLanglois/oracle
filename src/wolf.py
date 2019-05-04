import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

# disable beginning output
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, LSTM, GRU
from keras.preprocessing.sequence import TimeseriesGenerator

# disable beginning output
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# enable reproducibility
np.random.seed(0)

# read data
df = pd.read_csv('data/aapl.csv')[::-1]

# read and scale features, read response
x = StandardScaler().fit_transform(df[df.columns.difference(['Date', 'Correction'])].astype(float))
y = df['Correction'].values

# reshape as timeseries
series = TimeseriesGenerator(x, y, 14, batch_size=len(x))[0]
series_x = series[0]
series_y = series[1]

# split into train and test
train_x, test_x, train_y, test_y = train_test_split(series_x, series_y, test_size=0.1, shuffle=False)

# create model
model = Sequential()
model.add(LSTM(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', shuffle=False, metrics=['accuracy'])

# fit model 300 times
for i in range(1, 301):
	# fit model 1 epoch
	model.fit(train_x, train_y, epochs=1, verbose=0)

	# evaluate on train and test sets
	train_acc = model.evaluate(train_x, train_y, verbose=0)[1]
	test_acc = model.evaluate(test_x, test_y, verbose=0)[1]

	# output
	print('Epoch:', i)
	print('Train Acc:', round(train_acc, 2))
	print('Test Acc:', round(test_acc, 2), end='\n\n')
