import numpy as np
from keras.applications.vgg16 import VGG16
import keras
import h5py
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

dataset = np.load('../database.npz.npy')
#dataset = dataset[:12000]

labels = dataset[:, 1]


cat_map = categoreis(list(set(labels)))

dataset = np.array([[ item[0], cat_map[item[1]]] for item in dataset ])


img_row = 101
img_col = 101

np.random.shuffle(dataset)

N = np.shape(dataset)[0]
train_test_split_percentage = 0.75

X_train = dataset[:int(N * train_test_split_percentage), 0]
X_test = dataset[int(N * train_test_split_percentage):, 0]


X_train = np.array([x.reshape(img_row, img_col, 3) for x in X_train])
X_test = np.array([x.reshape(img_row, img_col, 3) for x in X_test])



y_train = dataset[:int(N * train_test_split_percentage), 1]
y_test = dataset[int(N * train_test_split_percentage):, 1]



y_train = keras.utils.to_categorical(y_train)
y_test  = keras.utils.to_categorical(y_test)


from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=(img_row,img_col,3),name = 'image_input')

import matplotlib.pyplot as plt


#Use the generated model
output_vgg16_conv = model_vgg16_conv(input)
print(np.shape(X_test), np.shape(y_test))
#Add the fully-connected layers
x = Flatten(name='flatten')(output_vgg16_conv)
x = keras.layers.BatchNormalization()(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(64, activation='relu', name='fc2')(x)
x = Dense(10, activation='softmax', name='predictions')(x)

#Create your own model
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
# my_model.summary()

my_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 64
epochs = 100
print(np.shape(y_train))
my_model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = my_model.evaluate(X_test, y_test, verbose=0)

# my_model.save()
my_model.save_weights('Training_Weights')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

