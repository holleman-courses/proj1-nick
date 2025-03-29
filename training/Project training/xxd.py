#!/usr/bin/env python

import argparse
import sys
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, layers
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
tfl_file_name = "85_small.tflite"
bytes_per_line = 12
indent = "  " # two spaces

num_calibration_steps = 25
model = tf.keras.models.load_model("85_small.h5")
model.summary()
# Replace MaxPooling2D layers with AveragePooling2D
for layer in model.layers:
    if isinstance(layer, MaxPooling2D):
        # Replace max pooling with average pooling
        model.layers[model.layers.index(layer)] = AveragePooling2D(pool_size=layer.pool_size,
                                                                    strides=layer.strides,
                                                                    padding=layer.padding)

# Save the modified model
model.save('modified_85_small.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_quant_model = converter.convert()

with open(tfl_file_name, "wb") as fpo:
  fpo.write(tflite_quant_model)
print(f"Wrote to {tfl_file_name}")
#!ls -l $tfl_file_name