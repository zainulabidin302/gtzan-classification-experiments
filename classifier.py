import numpy as np 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys

def E():
  sys.exit()

def Normalization(data):
    norm = data / np.linalg.norm(data)
    return norm

# np.random.seed(32)


def categoreis(labels):
	a = {}
	counter = 0
	for label in labels:
		a[label] = counter
		counter += 1
	return a

dataset = np.load('database-112x1-52037.npz.npy')
dataset = dataset[:]

labels = dataset[:, 1]


cat_map = categoreis(list(set(labels)))


# print("Before",dataset)

#dataset = np.array([ [ item[0][:, :20], cat_map[item[1]]] for item in dataset ])
dataset = np.array([ [ item[0], cat_map[item[1]]] for item in dataset ])

print(dataset[0][0])

print(np.shape(dataset))
print(np.shape(dataset[0]))


img_row = 112
img_col = 112
channels = 1




np.random.shuffle(dataset)

N = np.shape(dataset)[0]
train_test_split_percentage = 0.75 

print('BEFORE -->', np.shape(dataset))
X_train = dataset[:int(N * train_test_split_percentage), 0]
X_test = dataset[int(N * train_test_split_percentage):, 0]

# print(X_train)


X_train = np.array([x.reshape(img_row, img_col, channels) for x in X_train])
X_test = np.array([x.reshape(img_row, img_col, channels) for x in X_test])

# X_train = Normalization(X_train)
# X_test = Normalization(X_test)

# print(np.shape(X_train))
# print(X_train.reshape(np.shape(X_train)[0], 810, 810, 1))
# E()
# X_train = np.array(list(map(lambda x: np.array(x), X_train)))
# X_test = np.array(list(map(lambda x: np.array(x), X_test)))



y_train = dataset[:int(N * train_test_split_percentage), 1]
y_test = dataset[int(N * train_test_split_percentage):, 1]


print(y_train)
print(y_train.shape)


y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test  = keras.utils.to_categorical(y_test, num_classes=10)
# y_train = keras.utils.to_categorical(y_train, num_classes=10)
# y_test  = keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                  input_shape=(img_row,img_col, channels)))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 512
epochs = 10
print(np.shape(y_train))
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.33)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
import time
model.save('models/conv-8-custom_database-112x1-52037_{}.hd5'.format(int(time.time())))