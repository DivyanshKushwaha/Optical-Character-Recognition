import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Conv2D, MaxPooling2D,Flatten
from src.constants import *

def cnn_model():
    model = Sequential([
        Conv2D(32,(3,3), activation='relu',input_shape= (224,224,1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128,activation='relu'),
        Dense(len(allowed_units),activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

