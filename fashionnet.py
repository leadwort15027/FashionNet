from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt

input_shape = (228,228,3)
inputs = Input(shape=input_shape, name='inputs')
x = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(inputs)
x = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool1')(x)

x = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(x)
x = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool2')(x)

x = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(x)
x = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(x)
x = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool3')(x)

x = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(x)
conv4 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool4')(conv4)

x = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool5')(x)

x = Dense(1024, activation='relu', name='fc6_pose')(x)
outputs = Dense(1024, activation='relu', name='fc7_pose')(x)


model = Model(input=inputs, output=outputs)
model.summary()

plot(model, to_file='model.png')
