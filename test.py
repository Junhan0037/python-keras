from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=2, input_shape=(3,), activation="sigmoid"))
model.add(Dense(1))
model.summary()
