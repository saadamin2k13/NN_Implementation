import os
# import keras
# from keras import models
# from keras import backend
import numpy as np
import cv2
import tensorflow as tf
# from tensorflow import keras
# import pandas as pd
import matplotlib.pylab as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split,GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import preprocessing
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from dmba import plotDecisionTree

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=3)
# model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

