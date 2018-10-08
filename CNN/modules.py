# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import numpy as np

def conv_forward(x, w, b, conv_param):
    """
    a naive implementation of the forward pass for a convolutional layer
    """
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (H + 2 * pad - WW) / stride
    H_out = int(H_out)
    W_out = int(W_out)

    out = np.zeros((N, F, H_out, W_out))
    for n in range(N):
        conv_in = np.pad(x[n], ((0, 0), (pad, pad), (pad, pad)), mode='constant')
        for f in range(F):
            conv_w = w[f]
            conv_b = b[f]
            for i in range(H_out):
                for j in range(W_out):
                    conv_i = i * stride
                    conv_j = j * stride
                    conv_area = conv_in[:, conv_i : conv_i + HH, conv_j : conv_j + WW]
                    out[n, f, i, j] = np.sum(conv_area * conv_w) + conv_b

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward(dout, cache):
    """
    a naive implementation of the backward pass for a convolutional layer
    """
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (H + 2 * pad - WW) / stride
    H_out = int(H_out)
    W_out = int(W_out)

    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    for n in range(N):
        conv_in = np.pad(x[n], ((0, 0), (pad, pad), (pad, pad)), mode='constant')
        dconv_in = np.zeros(conv_in.shape)
        for f in range(F):
            conv_w = w[f]
            conv_b = b[f]
            df = dout[n, f]
            for i in range(H_out):
                for j in range(W_out):
                    conv_i = i * stride
                    conv_j = j * stride
                    conv_area = conv_in[:, conv_i : conv_i + HH, conv_j : conv_j + WW]
                    dconv = df[i, j]
                    db[f] += dconv
                    dw[f] += dconv * conv_area
                    dconv_in[:, conv_i: conv_i + HH, conv_j: conv_j + WW] += dconv * conv_w

    dx[n] += dconv_in[:, pad:-pad, pad:-pad]
    return dx, dw, db

def relu_forward(x):
    out = x * (x > 0)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    D = x[0].shape

    if mode == 'train':
        running_mean = np.zeros(D, dtype=x.dtype)
        running_var = np.zeros(D, dtype=x.dtype)

        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        sample_std = np.sqrt(sample_var + eps)

        x_norm = (x - sample_mean) / sample_std
        out = x_norm * gamma + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * sample_mean
        running_var = momentum * running_var + (1.0 - momentum) * sample_var

        cache = (x, sample_mean, sample_var, sample_std, x_norm, beta, gamma, eps)
    elif mode == 'test':
        running_mean = bn_param['running_mean']
        running_std = bn_param['running_std']
        x_norm = (x - running_mean) / running_std
        out = x_norm * gamma + beta
        cache = (running_mean, running_std)

    else:
        raise ValueError('Invalid forward batchnorm mode %s' % mode)

    # bn_param['running_mean'] = running_mean
    # bn_param['running_var'] = running_var
    return out, cache

def batchnorm_backward(dout, cache):
    (x, sample_mean, sample_var, sample_std, x_norm, beta, gamma, eps) = cache
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    N, D, HH, WW = x.shape
    dx_norm = dout * gamma
    dsample_var = np.sum(-1 / 2 * dx_norm * (x - sample_mean) / (sample_var + eps) ** (3 / 2), axis = 0)
    dsample_mean = np.sum(-1 / sample_std * dx_norm, axis=0) + 1 / N * dsample_var * np.sum(-2 * (x - sample_mean), axis=0)
    dx = 1 / sample_std * dx_norm + dsample_var * 2 / N * (x - sample_mean) + 1 / N * dsample_mean
    return dx, dgamma, dbeta

def dropout_forward(x, dp_param):
    p, mode = dp_param['p'], dp_param['mode']
    mask = None
    out = None
    if mode == 'train':
        mask = 1.0 * (np.random.rand(x.shape) > p)

def conv_relu_forward(x, w, b, conv_param):
    var, conv_cache = conv_forward(x, w, b, conv_param)
    out, relu_cache = rule_forward(var)
    cache = (conv_cache, relu_cache)
    return out, cache

def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    dvar = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward(dvar, conv_cache)
    return dx, dw, db

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    var1, conv_cache = conv_forward(x, w, b, conv_param)
    var2, bn_cache = batchnorm_forward(var1, gamma, beta, bn_param)
    out, relu_cache = relu_forward(var2)
    cache = (conv_cache, bn_cache, relu_cache)
    if bn_param['mode'] == 'train':
        x, mean, var, std, x_norm, beta, gamma, eps = bn_cache
        return out, cache, mean, std
    else:
        mean, std = bn_cache
        return out, cache, mean, std

def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dvar2 = relu_backward(dout, relu_cache)
    dvar1, dgamma, dbeta = batchnorm_backward(dvar2, bn_cache)
    dx, dw, db = conv_backward(dvar1, conv_cache)
    return dx, dw, db, dgamma, dbeta