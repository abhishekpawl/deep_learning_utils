import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0.01*Z, Z)
    return A

def sigmoid_backward(Z, dA):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)
    return dZ


def relu_backward(Z, dA):
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0.01

    assert (dZ.shape == Z.shape)
    return dZ


def initialize_parameters(layer_dims):
    W = {}
    b = {}

    for i in range(1, len(layer_dims)):
        W['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(1 / layer_dims[i - 1])
        b['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return W, b

def forward_propagation(layer_dims, A0, W, b, actfunc):
    Z = {}
    A = {}
    A["A0"] = A0

    for i in range(1, len(layer_dims)):
        Z['Z' + str(i)] = np.dot(W['W' + str(i)], A['A' + str(i - 1)]) + b['b' + str(i)]
        if actfunc[i - 1] == "sigmoid":
            A['A' + str(i)] = sigmoid(Z['Z' + str(i)])
        elif actfunc[i - 1] == "relu":
            A['A' + str(i)] = relu(Z['Z' + str(i)])

    return Z, A

def cost_function(AN, Y, m, epsilon = 1e-5):
    cost = -(1/m) * np.sum(np.multiply(Y, np.log(AN + epsilon)) + np.multiply((1-Y), np.log(1-AN + epsilon)))
    cost = np.squeeze(cost)
    return cost


def backward_propagation(layer_dims, A, Z, W, Y, actfunc, m):
    dA = {}
    dZ = {}
    dW = {}
    db = {}

    for i in range(len(layer_dims) - 1, 0, -1):
        if i == len(layer_dims) - 1:
            dA["dA" + str(i)] = A['A' + str(i)] - Y
        else:
            dA["dA" + str(i)] = np.dot(W['W' + str(i + 1)].T, dZ["dZ" + str(i + 1)])

        if actfunc[i - 1] == 'sigmoid':
            dZ["dZ" + str(i)] = sigmoid_backward(Z['Z' + str(i)], dA["dA" + str(i)])
        elif actfunc[i - 1] == 'relu':
            dZ["dZ" + str(i)] = relu_backward(Z['Z' + str(i)], dA["dA" + str(i)])

        dW["dW" + str(i)] = (1 / m) * np.dot(dZ["dZ" + str(i)], A['A' + str(i - 1)].T)
        db["db" + str(i)] = (1 / m) * np.sum(dZ["dZ" + str(i)], axis=1, keepdims=True)

    return dW, db, dZ["dZ1"]


def update_parameters(layer_dims, W, b, dW, db, learning_rate):
    for i in range(1, len(layer_dims)):
        W['W' + str(i)] = W['W' + str(i)] - learning_rate * dW["dW" + str(i)]
        b['b' + str(i)] = b['b' + str(i)] - learning_rate * db["db" + str(i)]

    return W, b
