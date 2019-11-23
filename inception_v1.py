import numpy as np
import keras
from keras.layers import Add,Conv2D,Dense,Dropout,Flatten,Activation,AveragePooling2D,MaxPooling2D,Input,LeakyReLU,BatchNormalization,ZeroPadding2D
# from keras.models import Model,Sequential,Layer
from keras.regularizers import l2
from keras.initializers import glorot_uniform
# from keras.applications.mobilenet import relu6,DepthwiseConv2D
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
import time

from functools import wraps
@wraps(Conv2D)
def my_conv(*args,**kwargs):
    new_kwargs={'kernel_regularizer':l2(1e-6)}
    new_kwargs['padding']='same'
    new_kwargs['kernel_size']=(1,1)
    new_kwargs['strides']=(2,2) if kwargs.get('strides')==(2,2) else (1,1)
    new_kwargs['kernel_initializer']=keras.initializers.glorot_uniform(seed=0)
    new_kwargs.update(kwargs)
    return Conv2D(*args,**new_kwargs)
def conv(x,**kwargs):
    x=my_conv(**kwargs)(x)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)
    return x

def inception_module(x,f1, f21, f22, f31, f32, f4, button=False):
    if button:
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # print('Max pool:', K.int_shape(x))
    x_branch1 = x_branch2 = x_branch3 = x_branch4 = x
    x_branch1 = conv(x_branch1, filters=f1)
    # print('x_branch1:', K.int_shape(x_branch1))

    x_branch2 = conv(x_branch2, filters=f21)
    x_branch2 = my_conv(filters=f22, kernel_size=(3, 3))(x_branch2)
    x_branch2 = BatchNormalization(axis=-1)(x_branch2)
    # print('x_branch2:', K.int_shape(x_branch2))

    x_branch3 = conv(x_branch3, filters=f31)
    x_branch3 = my_conv(filters=f32, kernel_size=(5, 5))(x_branch3)
    x_branch3 = BatchNormalization(axis=-1)(x_branch3)
    # print('x_branch3:', K.int_shape(x_branch3))

    x_branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_branch4)
    x_branch4 = my_conv(filters=f4)(x_branch4)
    x_branch4 = BatchNormalization(axis=-1)(x_branch4)
    # print('x_branch4:', K.int_shape(x_branch4))

    x = keras.layers.concatenate([x_branch1, x_branch2, x_branch3, x_branch4])
    x = LeakyReLU(alpha=0.05)(x)
    return x

def googlenet(input_shape, classes):
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(x_input)  # (None, 230, 230, 3)
    x = conv(x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')    # (None, 112, 112, 64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)   # (None, 56, 56, 64)
    x = BatchNormalization(axis=-1)(x)
    x = my_conv(filters=64)(x)  # (None, 56, 56, 64)
    x = BatchNormalization(axis=-1)(x)
    x = conv(x, filters=192, kernel_size=(3, 3))    # (None, 56, 56, 192)
    x = inception_module(x, 64, 96, 128, 16, 32, 32, button=True)   # (None, 28, 28, 256)
    x = inception_module(x, 128, 128, 192, 32, 96, 64, button=False) # (None, 28, 28, 480)
    x = inception_module(x, 192, 96, 208, 16, 48, 64, button=True)  # (None, 14, 14, 512)
    x = inception_module(x, 160, 112, 224, 24, 64, 64, button=False)  # (None, 14, 14, 512)
    x = inception_module(x, 128, 128, 256, 24, 64, 64, button=False)  # (None, 14, 14, 512)
    x = inception_module(x, 112, 144, 288, 32, 64, 64, button=False)  # (None, 14, 14, 528)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, button=False)  # (None, 14, 14, 832)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, button=True)   # (None, 7, 7, 832)
    x = inception_module(x, 384, 192, 384, 48, 128, 128, button=False)  # (None, 7, 7, 1024)
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(x)  # (None, 1, 1, 1024)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Flatten()(x)
    x = Dense(units=classes, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    # print('end:',K.int_shape(x))
    x = keras.models.Model(inputs=x_input, outputs=x)
    return x

if __name__ == '__main__':
    googlenet(input_shape=(224,224,3), classes=10)