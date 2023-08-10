#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tracarbon import (
    CarbonEmissionGenerator,
    EnergyConsumptionGenerator,
    StdoutExporter,
    TracarbonBuilder,
    TracarbonConfiguration,
    Location
)
import nest_asyncio
from tracarbon.locations import*

nest_asyncio.apply()

location=Country.get_location()
configuration = TracarbonConfiguration(co2signal_api_key="UALiba5FEs7d2negQ7CEVrEHeHjwpkLz")
metric_generators = [EnergyConsumptionGenerator(location=location), CarbonEmissionGenerator(location=location)]
exporter = StdoutExporter(metric_generators=metric_generators)
tracarbon = TracarbonBuilder().with_exporter(exporter=exporter).build()
with tracarbon: 
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # Define the model
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)

    # Convolutional block 1
    activation = 'relu'
    shortcut = x
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    shortcut = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(shortcut)# Adjusted shortcut
    x = layers.add([x, shortcut])
    x = layers.Activation(activation)(x)

    # Convolutional block 2
    shortcut = x
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    shortcut = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(shortcut)# Adjusted shortcut
    x = layers.add([x, shortcut])
    x = layers.Activation(activation)(x)

    # Convolutional block 3
    shortcut = x
    x = layers.Conv2D(filters=89, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters=89, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=89, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=89, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=89, kernel_size=3, strides=1, padding='same')(x)
    shortcut = layers.Conv2D(filters=89, kernel_size=3, strides=1, padding='same')(shortcut)# Adjusted shortcut
    x = layers.add([x, shortcut])
    x = layers.Activation(activation)(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully-connected layer
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(10, activation='relu')(x)


    # Softmax activation
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
report = tracarbon.report # to get the report
print(report)


# In[ ]:




