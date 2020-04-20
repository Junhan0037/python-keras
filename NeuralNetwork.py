# 임의로 선택할 값
selected_image = 0

# Keras Datasets Load from MNIST dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Import Keras Libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax')) # 0~9까지 분류 : 10종류

# Model Compile set up
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# preprocess our data by reshaping it into the shape between 0 and 1.
x_train = x_train.reshape((60000, 28 * 28)) # 1차원으로 늘리기
x_train = x_train.astype('float32') / 255 # between 0 and 1

x_test_original = x_test
x_test = x_test.reshape((10000, 28 * 28))
x_test = x_test.astype('float32') / 255

# To categorically encode the labels
from keras.utils import to_categorical
# 원핫코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# we "fit" the model to its training data.
model.fit(x_train, y_train, epochs=5, batch_size=128)

# Predict Digit
result = model.predict(np.array([x_test[selected_image]]))
result_number = np.argmax(result)

# Draw Digit Image
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 29, 5) # 굵은선
minor_ticks = np.arange(0, 29, 1) # 가는선

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

ax.imshow(x_test_original[selected_image], cmap=plt.cm.binary)

plt.show()

# Draw Result Number
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

digits = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
y_pos = np.arange(len(digits))
performance = [ val for val in result[0]]
print(performance)
result_probability = performance[result_number]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, digits)
plt.ylabel('probability')
plt.title('Number is %2i (probability %7.4f)' % (result_number, result_probability*100))

plt.show()
