#!/usr/bin/env python2.7
# Li Ding
# Mar. 2018


from __future__ import division

from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras import backend as K
from keras.optimizers import rmsprop


def dice_coeff(y_true, y_pred):
    smooth = 0.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def IOU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = intersection / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def sigmoid_cross_entropy(y_true, y_pred):
    z = K.flatten(y_true)
    x = K.flatten(y_pred)
    q = 10
    l = (1 + (q - 1) * z)
    loss = (K.sum((1 - z) * x) + K.sum(l * (K.log(1 + K.exp(- K.abs(x))) + K.max(-x, 0)))) / 500
    return loss


def GRU64(n_nodes, conv_len, n_classes, n_feat, in_len,
        optimizer=rmsprop(lr=1e-3), return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(in_len, n_feat))
    model = inputs

    model = CuDNNGRU(64, return_sequences=True)(model)
    model = SpatialDropout1D(0.5)(model)

    model.set_shape((None, in_len, 64))
    model = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(model)

    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode="temporal")

    if return_param_str:
        param_str = "GRU_C{}_L{}".format(conv_len, n_layers)
        return model, param_str
    else:
        return model


def TCFPN(n_nodes, conv_len, n_classes, n_feat, in_len,
          optimizer=rmsprop(lr=1e-4), return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(in_len, n_feat))
    model = inputs
    lyup = []
    lydown = []

    # ---- Encoder ----
    for i in range(n_layers):
        model = Conv1D(n_nodes[i], conv_len, padding='same', use_bias=False)(model)
        model = BatchNormalization()(model)
        model = SpatialDropout1D(0.1)(model)
        model = Activation('relu')(model)
        model = MaxPooling1D(2, padding='same')(model)
        lyup.append(model)

    # ---- Decoder ----
    model = Conv1D(n_nodes[0], 1, padding='same', use_bias=False)(model)
    modelout = SpatialDropout1D(0.1)(model)
    modelout = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(modelout)
    modelout = UpSampling1D(8)(modelout)
    lydown.append(modelout)

    model = UpSampling1D(2)(model)
    res = Conv1D(n_nodes[0], 1, padding='same', use_bias=False)(lyup[-2])
    model = add([model, res])
    model = Conv1D(n_nodes[0], conv_len, padding='same', use_bias=False)(model)
    modelout = SpatialDropout1D(0.1)(model)
    modelout = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(modelout)
    modelout = UpSampling1D(4)(modelout)
    lydown.append(modelout)

    model = UpSampling1D(2)(model)
    res = Conv1D(n_nodes[0], 1, padding='same', use_bias=False)(lyup[-3])
    model = add([model, res])
    model = Conv1D(n_nodes[0], conv_len, padding='same', use_bias=False)(model)
    modelout = SpatialDropout1D(0.1)(model)
    modelout = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(modelout)
    modelout = UpSampling1D(2)(modelout)
    lydown.append(modelout)

    model = UpSampling1D(2)(model)
    res = Conv1D(n_nodes[0], 1, padding='same', use_bias=False)(inputs)
    model = add([model, res])
    model = Conv1D(n_nodes[0], conv_len, padding='same', use_bias=False)(model)
    modelout = SpatialDropout1D(0.1)(model)
    modelout = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(modelout)
    lydown.append(modelout)

    model = average(lydown)

    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  sample_weight_mode="temporal")

    if return_param_str:
        param_str = "TCFPN_C{}_L{}".format(conv_len, n_layers)
        return model, param_str
    else:
        return model
