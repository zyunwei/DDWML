import numpy
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

with open('herolist.json', 'r', encoding='utf-8') as f:
    hero_data = json.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dim = 126

hero_pool = np.genfromtxt("heropool.csv", skip_header=1, delimiter=',', dtype=int)
hero_wins = np.genfromtxt("herowins.csv", skip_header=1, delimiter=',', dtype=int)
data_length = hero_pool.shape[0]
sample_in = []
sample_out = []

for data_index, line in enumerate(hero_pool):
    for hero_index, is_exist in enumerate(hero_pool[data_index]):
        if is_exist == 1:
            hero_select = np.zeros(dim)
            hero_select[hero_index] = 1
            sample_in.append([line, hero_select])
            if hero_wins[data_index, hero_index] == 1:
                sample_out.append(1.0)
            else:
                sample_out.append(0.0)

sample_in = numpy.array(sample_in)
sample_out = numpy.array(sample_out)

x_train, x_test, y_train, y_test = train_test_split(sample_in, sample_out, test_size=0.1, shuffle=True)

model = Sequential()
model.add(LSTM(256, input_shape=(2, dim), return_sequences=False))
model.add(Dense(10, activation="sigmoid"))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

print("fit:")
model.fit(x_train, y_train, epochs=5, batch_size=512)

print("eval:")
mae = model.evaluate(x_test, y_test, verbose=2)
print(mae)

model.save('saved_model/20250101_20250201.h5', save_format='h5')
