import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import utils as utls
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from sklearn.metrics import classification_report

imageRows, imageCols = 32, 32
batchSize = 256
numClasses = 10
epochs = 10

(XTrain, yTrain), (XTest, yTest) = cifar10.load_data()

XTrain = XTrain.astype('float32')
XTest = XTest.astype('float32')
XTrain = XTrain / 255.0
XTest = XTest / 255.0

yTrain = utls.to_categorical(yTrain, numClasses)
yTest = utls.to_categorical(yTest, numClasses)
inputShape = (imageRows, imageCols, 3)

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(50, (5,5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(numClasses))
model.add(Activation('softmax'))
model.summary()

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
minhaLeNetModel = model.fit(XTrain, yTrain, batch_size=batchSize, epochs=epochs, validation_data=(XTest, yTest))

previsto= model.predict(XTest)

esperado_classes = np.argmax(yTest, axis=1)
previsto_classes = np.argmax(previsto, axis=1)


nome = ["Avião", "Automóvel", "Pássaro", "Gato", "Veado", "Cachorro", "Sapo", "Cavalo", "Navio", "Caminhão"]
print(classification_report(esperado_classes, previsto_classes, target_names=nome))
