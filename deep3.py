from keras.layers import Concatenate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from keras import utils as utls
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report

imageRows, imageCols = 32, 32
numClasses = 10
inputShape = (imageRows, imageCols, 3)

def inception_block(x, f1, f3, f5):
    conv1 = Conv2D(f1, (1,1), activation='relu', padding='same')(x)
    conv3 = Conv2D(f3, (3,3), activation='relu', padding='same')(x)
    conv5 = Conv2D(f5, (5,5), activation='relu', padding='same')(x)
    pool = MaxPooling2D((3,3), strides=1, padding='same')(x)
    
    return Concatenate()([conv1, conv3, conv5, pool])


(XTrain, yTrain), (XTest, yTest) = cifar10.load_data()

XTrain = XTrain.astype("float32") / 255.0
XTest = XTest.astype("float32") / 255.0

yTrain_cat = utls.to_categorical(yTrain, numClasses)
yTest_cat = utls.to_categorical(yTest, numClasses)

inputs = tf.keras.Input(shape=inputShape)

x = inception_block(inputs, 32, 32, 32)
x = MaxPooling2D((2,2))(x)

x = inception_block(x, 64, 64, 64)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(numClasses, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    XTrain, yTrain_cat,
    validation_data=(XTest, yTest_cat),
    batch_size=128,
    epochs=10
)

pred_probs = model.predict(XTest)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = yTest.flatten()
print(classification_report(true_classes, pred_classes))

