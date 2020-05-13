import numpy as np

def initialize_parameters_conv():
    np.random.seed(1)
    W = np.random.randn(3, 3, 3, 8)
    b = np.zeros((1, 1, 1, 8))
    return W, b

def pad_with_zero(X, pad):
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode = 'constant', constant_values = 0)
    return X

def convolution(window, W, b):
    s = np.multiply(window, W)
    conv = np.sum(s, axis = None)
    conv = float(conv + b)
    return conv

def convolution_layer(A_prev, W, b, hparams):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hparams['stride']
    pad = hparams['pad']

    n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

    convd_A = np.zeros((m, n_H, n_W, n_C))

    padded_A_prev = pad_with_zero(A_prev, pad)

    for i in range(m):
        a = padded_A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    window = a[vert_start:vert_end, horiz_start:horiz_end, :]
                    convd_A[i, h, w, c] = convolution(window, W[:, :, :, c], b[:, :, :, c])

    assert(convd_A.shape == (m, n_H, n_W, n_C))

    return convd_A

def pooling_layer(A_prev, hparams):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = 2
    stride = hparams["stride"]

    n_H = int((n_H_prev - f) / stride + 1)
    n_W = int((n_W_prev - f) / stride + 1)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    window = A_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    A[i, h, w, c] = np.max(window)

    assert(A.shape == (m, n_H, n_W, n_C))

    return A

def conv_back_propagation(dZ, A_prev, W, b, hparams):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparams['stride']
    pad = hparams['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    padded_A_prev = pad_with_zero(A_prev, pad)
    padded_dA_prev = pad_with_zero(dA_prev, pad)

    for i in range(m):
        padded_a_prev = padded_A_prev[i]
        padded_da_prev = padded_dA_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    window = padded_a_prev[vert_start:vert_end, horiz_start:horiz_end, ]

                    padded_da_prev[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += np.multiply(window, dZ[i, h, w, c])
                    db = np.sum(dZ, axis = (0, 1, 2))

        dA_prev[i, :, :, :] = padded_da_prev[pad:-pad, pad:-pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

def update_parameters_conv(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b
