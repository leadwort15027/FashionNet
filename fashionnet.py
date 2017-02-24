from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt

input_shape = (224,224,3)
inputs = Input(shape=input_shape, name='inputs')
x = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(inputs)
x = ZeroPadding2D(padding=(1,1), name='pad1_1')(x)
x = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(x)
x = ZeroPadding2D(padding=(1,1), name='pad1_2')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool1')(x)

x = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(x)
x = ZeroPadding2D(padding=(1,1), name='pad2_1')(x)
x = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(x)
x = ZeroPadding2D(padding=(1,1), name='pad2_2')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool2')(x)

x = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(x)
x = ZeroPadding2D(padding=(1,1), name='pad3_1')(x)
x = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(x)
x = ZeroPadding2D(padding=(1,1), name='pad3_2')(x)
x = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(x)
x = ZeroPadding2D(padding=(1,1), name='pad3_3')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool3')(x)

x = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(x)
x = ZeroPadding2D(padding=(1,1), name='pad4_1')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(x)
x = ZeroPadding2D(padding=(1,1), name='pad4_2')(x)
x = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(x)
pad4_3 = ZeroPadding2D(padding=(1,1), name='pad4_3')(x)
# pad4_3 -> conv5_local

# POSE
p = MaxPooling2D(pool_size=(2,2), name='pool4_pose')(pad4_3)
p = Convolution2D(512, 3, 3, activation='relu', name='conv5_pose_1')(p)
p = ZeroPadding2D(padding=(1,1), name='pad5_pose_1')(p)
p = Convolution2D(512, 3, 3, activation='relu', name='conv5_pose_2')(p)
p = ZeroPadding2D(padding=(1,1), name='pad5_pose_2')(p)
p = Convolution2D(512, 3, 3, activation='relu', name='conv5_pose_3')(p)
p = ZeroPadding2D(padding=(1,1), name='pad5_pose_3')(p)
p = MaxPooling2D(pool_size=(2,2), name='pool5_pose')(p)
p = Flatten()(p)
p = Dense(1024, activation='relu', name='fc6_pose')(p)
p = Dense(1024, activation='relu', name='fc7_pose')(p)
loc = Dense(8, activation='linear',name='location')(p)
vis1 = Dense(2, activation='softmax', name='visibility1')(p)
vis2 = Dense(2, activation='softmax', name='visibility2')(p)
vis3 = Dense(2, activation='softmax', name='visibility3')(p)
vis4 = Dense(2, activation='softmax', name='visibility4')(p)

# LOCAL


# GLOBAL
g = MaxPooling2D(pool_size=(2,2), name='pool4_global')(pad4_3)
g = Convolution2D(512, 3, 3, activation='relu', name='conv5_global_1')(g)
g = ZeroPadding2D(padding=(1,1), name='pad5_global_1')(g)
g = Convolution2D(512, 3, 3, activation='relu', name='conv5_global_2')(g)
g = ZeroPadding2D(padding=(1,1), name='pad5_global_2')(g)
g = Convolution2D(512, 3, 3, activation='relu', name='conv5_global_3')(g)
g = ZeroPadding2D(padding=(1,1), name='pad5_global_3')(g)
g = MaxPooling2D(pool_size=(2,2), name='pool5_global')(g)
g = Flatten()(g)
fc6_global = Dense(4096, activation='relu', name='fc6_global')(g)


outputs=[loc,vis1,vis2,vis3,vis4]

model = Model(input=inputs, output=outputs)
model.summary()

plot(model, to_file='model.png')
