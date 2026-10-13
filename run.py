import cv2
import numpy as np

arr = cv2.imread('digit_28x28.png')     # arr.shape = (28,28,3)
arr = np.mean(arr,axis = 2)    # axis = 2 -> 3rd dimension
arr = arr.reshape(784)

saved_data = np.load('trained_nn_data.npz')

w1 = saved_data['w1']
w2 = saved_data['w2']
w3 = saved_data['w3']
b1 = saved_data['b1']
b2 = saved_data['b2']
b3 = saved_data['b3']

print("Data imported successfully.")

def relu(x):
    return np.maximum(0,x)

def relz(x):
    return x - np.max(x)    # Relative x

def sm(x,expsum):
    return np.exp(x)/expsum

def loss(x,op):
    return (x-op)**2

anet = np.zeros((1,10))

x = arr

z1 = x @ w1 + b1
a1 = relu(z1)

z2 = a1@w2 + b2
a2 = relu(z2)

z3 = a2@w3 + b3
relz3 = relz(z3)
expsum = np.sum(np.exp(relz3))
a3 = sm(relz3, expsum)

print("the number is ",np.argmax(a3))   # argmax : used to get the index of maximum number in array
print("Accuracy : ",np.max(a3)*100,"%")