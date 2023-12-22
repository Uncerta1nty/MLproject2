import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle

#DELETE
import matplotlib.pyplot as plt
import time

def initializeWeights(n_in, n_out):

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):

    return  1.0 / (1.0 + np.exp(-z))


def preprocess():
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    # Your code here.

    x = np.vstack([mat['train' + str(i)] for i in range(10)])
    y = np.hstack([i * np.ones(6000) for i in range(10)])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    validation_indices = indices[:10000]
    train_indices = indices[10000:60000]

    z = x[validation_indices, :]
    w = y[validation_indices]

    x = x[train_indices, :]
    y = y[train_indices]

    non_repetitive_features = np.where(np.std(x, axis=0) != 0)[0]

    x = x[:, non_repetitive_features]
    z = z[:, non_repetitive_features]

    test_data = np.vstack([mat['test' + str(i)] for i in range(10)])
    test_label = np.hstack([i * np.ones(1000) for i in range(10)])

    test_data = test_data[:, non_repetitive_features]
    print('preprocess done')

    return x, y, z, w, test_data, test_label


def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    n = training_data.shape[0]
    training_data = np.column_stack((np.ones((n, 1)), training_data))
    a = np.dot(training_data, w1.T)
    z = sigmoid(a)
    z = np.column_stack((np.ones((n, 1)), z))
    b = np.dot(z, w2.T)
    o = sigmoid(b)
    y = np.zeros((n, n_class))
    for i in range(n):
        y[i, int(training_label[i])] = 1
    J = -1 / n * np.sum(y * np.log(o) + (1 - y) * np.log(1 - o))
    reg_term = lambdaval / (2 * n) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))
    obj_val = J + reg_term
    delta2 = o - y
    grad_w2 = 1 / n * np.dot(delta2.T, z) + lambdaval / n * np.column_stack((np.zeros((n_class, 1)), w2[:, 1:]))
    delta1 = np.dot(delta2, w2) * z * (1 - z)
    grad_w1 = 1 / n * np.dot(delta1[:, 1:].T, training_data) + lambdaval / n * np.column_stack((np.zeros((n_hidden, 1)), w1[:, 1:]))
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()))
    return (obj_val, obj_grad)


def nnPredict(x, y, data):

    a = np.hstack((np.ones((data.shape[0], 1)), data))
    b = np.dot(a, x.T)
    c = sigmoid(b)
    c = np.hstack((np.ones((c.shape[0], 1)), c))
    d = np.dot(c, y.T)
    e = sigmoid(d)

    labels = np.argmax(e, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

 # Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 40

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10
#
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


predicted_label = nnPredict(w1, w2, train_data)




params = {'w1': w1, 'w2': w2, 'lambdaval': lambdaval, 'n_hidden': n_hidden}
with open('params.pickle', 'wb') as file:
    pickle.dump(params, file)