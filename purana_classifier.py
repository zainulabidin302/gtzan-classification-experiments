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

np.random.seed(32)


def categoreis(labels):
	a = {}
	counter = 1
	for label in labels:
		a[label] = counter
		counter += 1
	return a

dataset = np.load('database.npz.npy')
print(np.shape(dataset))

labels = dataset[:, 1]



cat_map = categoreis(list(set(labels)))


#dataset = np.array([ [ item[0][:, :20], cat_map[item[1]]] for item in dataset ])
dataset = np.array([ [ item[0], cat_map[item[1]]] for item in dataset ])

# img_row = np.shape(dataset[0][0])[0]
# img_col = np.shape(dataset[0][0])[1]
img_row = 810
img_col = 810

np.random.shuffle(dataset)

N = np.shape(dataset)[0]
train_test_split_percentage = 0.75 

print('BEFORE -->', np.shape(dataset))
X_train = dataset[:int(N * train_test_split_percentage), 0]
X_test = dataset[int(N * train_test_split_percentage):, 0]


# X_train = np.array(list(map(lambda x: np.array(x), X_train)))
# X_test = np.array(list(map(lambda x: np.array(x), X_test)))
print(X_train.shape, X_train[0].shape)
print(X_train.reshape((750, 1)))
np.array(X_train, (1, 810, 810))
print(np.shape(X_train), np.shape(X_test))
E()
X_train = X_train.reshape(X_train.shape[0], img_row, img_col, 1)
X_test = X_test.reshape(X_test.shape[0], img_row, img_col, 1)

input_shape = (img_row, img_col, 1)

y_train = dataset[:int(N * train_test_split_percentage), 1]
y_test = dataset[int(N * train_test_split_percentage):, 1]
y_train = keras.utils.to_categorical(y_train)
y_test  = keras.utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                  input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(cat_map.keys())+1, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 28
epochs = 10
print(np.shape(y_train))
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])