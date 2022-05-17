
import pickle

import keras

import similaritymeasures as sm
from tensorflow.keras import backend as K

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Bidirectional, LSTM, Flatten, Dense, Reshape, UpSampling1D, TimeDistributed
from tensorflow.keras.layers import Activation, Conv1D, LeakyReLU, Dropout, Add, Layer
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM as CUDNNLSTM
from tensorflow.keras.optimizers import Adam

from functools import partial
from scipy import integrate, stats


class RandomWeightedAverage(Layer):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 4))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def build_encoder_layer(input_shape, encoder_reshape_shape):    
    
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Bidirectional(LSTM(units=100, return_sequences=True))(input_layer)
    x = layers.Flatten()(x)
    x = layers.Dense(encoder_reshape_shape[0]*encoder_reshape_shape[1])(x)
    x = layers.Reshape(target_shape=encoder_reshape_shape)(x)
    model = keras.models.Model(input_layer, x, name='encoder')
    
    return model

def build_generator_layer(input_shape, generator_reshape_shape):
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Flatten()(input_layer)
    x = layers.Dense(generator_reshape_shape[0]*generator_reshape_shape[1])(x)
    x = layers.Reshape(target_shape=generator_reshape_shape)(x)
    x = layers.Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(x)
    x = layers.TimeDistributed(layers.Dense(4))(x)
    x = layers.Activation(activation='tanh')(x)
    model = keras.models.Model(input_layer, x, name='generator')
    
    return model
    

def build_critic_x_layer(input_shape):
    
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(filters=64, kernel_size=5)(input_layer)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.25)(x)
    x = layers.Conv1D(filters=64, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.25)(x)
    x = layers.Conv1D(filters=64, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.25)(x)
    x = layers.Conv1D(filters=64, kernel_size=5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=1)(x)
    model = keras.models.Model(input_layer, x, name='critic_x')
    
    return model 


def build_critic_z_layer(input_shape):
    
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Flatten()(input_layer)
    x = layers.Dense(units=100)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.2)(x)    
    x = layers.Dense(units=100)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.2)(x)  
    x = layers.Dense(units=1)(x)
    model = keras.models.Model(input_layer, x, name='critic_z')
    
    return model

def wasserstein_loss(y_true, y_pred):
#    return tf.reduce_mean(y_true * y_pred)
    return K.mean(y_true * y_pred)