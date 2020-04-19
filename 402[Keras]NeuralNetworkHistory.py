selected_image = 0

# Keras Datasets Load from MNIST dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.___________

# Import Keras Libraries
from keras.models import _________
from keras.layers import ______
import numpy as np

model = ________
model.add(Dense(512, activation='relu', _________=(28 * 28,)))
model.add(Dense(10, activation='________'))

# Model Compile set up
model.________(optimizer='rmsprop',
                loss='_________________',
                metrics=[_________])

# preprocess our data by reshaping it into the shape between 0 and 1.
x_train = x_train.________((60000, 28 * 28))
x_train = x_train._______('float32') / 255

x_test_original = x_test
x_test = x_test._________((10000, 28 * 28))
x_test = x_test.______('float32') / 255

# To categorically encode the labels
from keras.utils import ____________

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# we "fit" the model to its training data.
history = model.___(x_train, y_train, 
                    ____________=(x_test, y_test),
                    _____=5, batch_size=128)

"""
SHow history
"""
print('\nAccuracy: {:.4f}'.format(model._______(x_test, y_test)[1]))
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
fig, acc_ax = plt.subplots()

loss_ax.plot(history.history[_____], 'ro', label='train loss')
loss_ax.plot(history.history[_________], 'r:', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history[_____], 'bo', label='train acc')
acc_ax.plot(history.history[________], 'b:', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

# Predict Digit
result = model.______(np.array([x_test[__________]]))
result_number = np.______(result)

# Draw Digit Image
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 29, 5)
minor_ticks = np.arange(0, 29, 1)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax._____(which='both')

# Or if you want different settings for the grids:
ax.____(which='minor', alpha=0.2)
ax.____(which='major', alpha=0.5)

ax.imshow(x_test_original[___________], cmap=plt.cm.binary)

plt.show()

# Draw Result Number
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

digits = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
y_pos = np.arange(___(digits))
performance = [ val for val in result[0]]
print(performance)
result_probability = performance[__________]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, digits)
plt.ylabel('probability')
plt.title('Number is %2i (probability %7.4f)' % (result_number, result_probability*100))

plt.show()
