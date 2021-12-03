# An end to end Artificial Neural Network model on IRIS data

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from sklearn.datasets import load_iris
import pandas as pd

data = pd.read_csv("Iris.csv")
Y = np.array(data["user_action"])
X = data.drop("user_action", axis = 1)
X = X.T
Y = Y.reshape((1,10))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def layer_sizes(X, Y):
    n_x = X.shape[0] 
    n_h = 4
    n_y = Y.shape[0] 
    
    return (n_x, n_h, n_y)
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    
  
    W1 = np.random.randn(n_h, n_x) 
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 =parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
def compute_cost(A2, Y, parameters):
    
    
    m = Y.shape[1] 

    
    logprobs = np.multiply(np.log(A2),Y)
    cost =  np.sum(logprobs) * (-1)
    
    
    cost = float(np.squeeze(cost))
    
    
    return cost

def backward_propagation(parameters, cache, X, Y):
   
    m = X.shape[1] 
    W1 = parameters["W1"]
    W2 = parameters["W2"]
   
    
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
   
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
 
    
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
  
    W1 = W1 - dW1 * learning_rate
    b1 = b1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    b2 = b2 - db2 * learning_rate
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False, learning_rate = 0.01):
    
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
         
    
        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(A2, Y, parameters)
 
        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 >= 0.5).astype(np.int).reshape((1,10))
    return predictions
from sklearn.metrics import f1_score
scores = []
from sklearn.pipeline import make_pipeline
for i in range(1,8):
    learning_rate = 0.1
    for j in range(0,10):
        parameters = nn_model(X, Y, n_h = i, num_iterations = 1000, print_cost = False, learning_rate = j)
        y_pred = predict(parameters, X)
        learning_rate += 0.1
        scores.append((i,j,f1_score(Y.T, y_pred.T)))
