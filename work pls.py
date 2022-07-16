import tensorflow as tf
import keras
from keras.callbacks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *

from matplotlib import image
from matplotlib import pyplot

from PIL import Image

from mpu import ml


dfTrain = pd.read_csv('sign_mnist_train.csv')
dfTest = pd.read_csv('sign_mnist_test.csv')


# processing data to make it suitable to feed into our CNN
yTrainArray = dfTrain['label'].to_numpy(dtype='uint8')
yTrain = ml.indices2one_hot(yTrainArray, nb_classes=26)
for row in yTrain:
    for i in range(0, len(row)):
        row[i] = float(row[i])
yTrainFinal = np.array(yTrain)

dfData = dfTrain.drop(['label'], axis = 1)
xTrainArray = dfData.to_numpy(dtype='uint8')/255
xTrainFinal = xTrainArray.reshape(27455, 28, 28, 1)
#print(xTrainFinal)
yTestArray = dfTest['label'].to_numpy(dtype='uint8')
yTest = ml.indices2one_hot(yTestArray, nb_classes=26)
for row in yTest:
    for i in range(0, len(row)):
        row[i] = float(row[i])
yTestFinal = np.array(yTest)

dfTestData = dfTest.drop(['label'], axis = 1)
xTestArray = dfTestData.to_numpy(dtype='uint8')/255
xTestFinal = xTestArray.reshape(7172, 28, 28, 1)

#print(xTestFinal)







#
#
# model = tf.keras.models.Sequential()
#
# model.add(Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
# model.add(Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
#
# model.add(MaxPooling2D(2,2))
# model.add(BatchNormalization())
#
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64,(3,3),activation="relu"))
# model.add(Conv2D(64,(3,3),activation="relu"))
# model.add(MaxPooling2D(2,2))
# model.add(BatchNormalization())
#
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(128,activation="relu"))
#
# model.add(Dense(26,activation="softmax"))
#
# class myCallback(Callback):
#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('val_accuracy')>0.995):
#       print("\nReached 99.5% accuracy so cancelling training!")
#       self.model.stop_training = True
# callback=myCallback()
#
# dynamicrate = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
#
#
# model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
# model.fit(xTrainFinal,yTrainFinal,epochs=4,validation_data=(xTestFinal, yTestFinal), callbacks=[callback,dynamicrate])




#model.save('cnn.h5')

# test_loss, test_acc = model.evaluate(xTestFinal, yTestFinal)
# print("Test Loss: {0} - Test Acc: {1}".format(test_loss, test_acc))
