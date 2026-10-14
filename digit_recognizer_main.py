# Go to draw.py , run it and save.
# Then go to run.py and run it.

# The training data is inside the 'Others' directory named as 'mnist_train.csv'. Take it out in same directory as 'digit_recognizer_main.py' is if you want to train it yourself.

import random
import numpy as np
import pandas as pd

data = pd.read_csv("mnist_train.csv")
# Because pandas data read and write speed is slow, converting data in np array in early stage is benifitial with respect to time.
data_np = data.to_numpy(dtype=np.float32)
data_np[ : , 1:785] /= 255.0     # Prevents the label from being devided by 255.0

print("Data successfully imported.")

# Learning Rate :
n = 0.001

def relu(x):
    return np.maximum(0,x)
    
# Differentiation of ReLu
def adash(x):
    return (x>0)

# Updated to prevent overflow (if z becomes large then the e^(large) will become too large. so it will be overflowed and will return not a number(NaN)) and better numerical stability
# Updated for numerical stability (prevents NaN errors)
def relz(x):
    return x - np.max(x)    # Relative x

# SoftMax function
def sm(x,expsum):
    return np.exp(x)/expsum
# Note : Differentiation of SoftMax is 1 (because of cross entropy)

# Variable initialization

w1 = np.random.uniform(-0.1,0.1,(784,16))
b1 = np.random.uniform(-0.1,0.1,(1,16))

w2 = np.random.uniform(-0.1,0.1,(16,16))
b2 = np.random.uniform(-0.1,0.1,(1,16))

w3 = np.random.uniform(-0.1,0.1,(16,10))
b3 = np.random.uniform(-0.1,0.1,(1,10))

print("All variables successfully initialized. Training is starting...")

# epochs :
for epoch in range(10):
    for i in range (data_np.shape[0]) :     # The pandas consider forst row as headers. so it will consider only 2 to 60000 (for excel) or 0 to 59998 (for indexing). so, we do this instead pf fixing the loop counts.
        x = data_np[i,1:785].reshape(1,784)
        op = np.zeros((10))
        op1 = int(data_np[i,0])    # it returns a single value
        op[op1] = 1
        op = op.reshape(1,10)

        z1 = x @ w1 + b1
        a1 = relu(z1)

        z2 = a1@w2 + b2
        a2 = relu(z2)

        z3 = a2@w3 + b3
        relz3 = relz(z3)    #relative z3 specially for SoftMax fxn overload preventing
        expsum = np.sum(np.exp(relz3))
        a3 = sm(relz3, expsum)

        # Here w1,w2 and w3 are old matrix
        d3 = a3 - op
        d2 = d3@w3.T * adash(z2)
        d1 = d2@w2.T * adash(z1)

        w3 -= n*(a2.T@d3)
        b3 -= n*d3

        w2 -= n*(a1.T@d2)
        b2 -= n*d2

        w1 -= n*(x.T@d1)
        b1 -= n*d1


    if((0<=epoch and epoch <= 10) or (10<epoch and epoch < 100 and epoch % 10 == 0) or (epoch % 100 == 0)) :
        print("Epoch ",epoch+1," completed.")
print("Training completed...")


# Save all weights and biases into a single file named 'mnist_weights.npz'
np.savez('trained_nn_data.npz', w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

print("Weights successfully saved to 'trained_nn_data.npz'!")
