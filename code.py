import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

#import matplotlib.pyplot as plt
#import app

batch = 32

generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_iterator = generator.flow_from_directory('data/train',batch_size = batch,color_mode='grayscale', class_mode='sparse')
test_iterator = generator.flow_from_directory('data/test',batch_size = batch,color_mode='grayscale', class_mode='sparse')

def design_model():
  model = Sequential()
  model.add(tf.keras.Input(shape=(256, 256, 1)))

  model.add(tf.keras.layers.Conv2D(2, 5, strides=3, activation="relu")) 
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5,5)))
  model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu")) 
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,2), strides=(2,2)))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(4, activation="softmax"))

  model.summary()

  callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

  return model, callback
  print("Model designed")

model, callback = design_model()

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
model.fit(train_iterator, epochs=50, steps_per_epoch=125, validation_data=test_iterator,validation_steps=100, callbacks=[callback], verbose=1)
