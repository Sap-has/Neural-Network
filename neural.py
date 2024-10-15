import numpy as np
# Creating data set

#Add all letters
# A
a = [0, 0, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1]
# B
b = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0]
# C
c = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 0]
# D
d = [0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 0, 0]
# E
e = [0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 0, 0]
# F
f = [0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0]
# G
g = [0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 0, 0]
# H
h = [0, 1, 0, 0, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 0, 0, 1, 0]
# I
i = [0, 0, 1, 1, 1, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 1, 1, 1, 0]
# J
j = [0, 0, 0, 0, 1, 0,
     0, 0, 0, 0, 1, 0,
     0, 0, 0, 0, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 0, 1, 1, 0, 0]
# K
k = [0, 1, 0, 0, 1, 0,
     0, 1, 0, 1, 0, 0,
     0, 1, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 1, 0, 0, 1, 0]
# L
l = [0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 0]
# M
m = [1, 0, 0, 0, 0, 1,
     1, 1, 0, 0, 1, 1,
     1, 0, 1, 1, 0, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1]
# N
n = [1, 0, 0, 0, 1, 0,
     1, 1, 0, 0, 1, 0,
     1, 0, 1, 0, 1, 0,
     1, 0, 0, 1, 1, 0,
     1, 0, 0, 0, 1, 0]
# O
o = [0, 1, 1, 1, 0, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     0, 1, 1, 1, 0, 0]
# P
p = [0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0]
# Q
q = [0, 1, 1, 1, 0, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 1, 0, 1, 0,
     0, 1, 1, 1, 1, 0]
# R
r = [0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 0, 0,
     0, 1, 0, 1, 0, 0,
     0, 1, 0, 0, 1, 0]
# S
s = [0, 1, 1, 1, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 0, 0,
     0, 0, 0, 0, 1, 0,
     0, 1, 1, 1, 0, 0]
# T
t = [1, 1, 1, 1, 1, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 0, 1, 0, 0]
# U
u = [1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     0, 1, 1, 1, 0, 0]
# V
v = [1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 1, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0]
# W
w = [1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 1, 1, 0, 1,
     0, 1, 0, 0, 1, 0]
# X
x = [1, 0, 0, 0, 1, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 1, 0, 1, 0, 0,
     1, 0, 0, 0, 1, 0]
# Y
y = [1, 0, 0, 0, 1, 0,
     0, 1, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 0, 1, 0, 0, 0]
# Z
z = [1, 1, 1, 1, 1, 0,
     0, 0, 0, 1, 0, 0,
     0, 0, 1, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 0]

 
# Creating labels
labels = np.eye(26).tolist()

letters = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]

# Convert binary data to numpy arrays and reshape
x = [np.array(letter).reshape(1, 30) for letter in letters]

# Convert labels to numpy arrays
y = np.array(labels)

import matplotlib.pyplot as plt

# visualizing the data, plotting A.
#plt.imshow(np.array(a).reshape(5, 6))
#plt.show()


# converting data and labels into numpy array
"""
Convert the matrix of 0 and 1 into one hot vector 
so that we can directly feed it to the neural network,
these vectors are then stored in a list x.
"""
 
 
print(x, "\n\n", y)


def relu(x):
    return np.maximum(0, x)

# activation function
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

# Creating the Feed forward neural network
# 1 Input layer(1, 30)
# 1 hidden layer (1, 5)
# 1 output layer(3, 3)
 
def f_forward(x, w1, w2):
   # hidden
   z1 = x.dot(w1)  # Input to hidden layer
   a1 = relu(z1)  # Activation from hidden layer
   
   z2 = a1.dot(w2)  # Input to output layer
   a2 = sigmoid(z2)  # Output from output layer
   return(a2)
  
# initializing the weights randomly
def generate_wt(x, y):
    l =[]
    for i in range(x * y):
        l.append(np.random.randn())
    return(np.array(l).reshape(x, y))
 
# for loss we will be using mean square error(MSE)
def loss(out, Y):
    s =(np.square(out-Y))
    s = np.sum(s)/len(y)
    return(s)

# Back propagation of error 
def back_prop(x, y, w1, w2, alpha):
    z1 = x.dot(w1)  # Input from layer 1
    a1 = relu(z1)  # Output of layer 2
    z2 = a1.dot(w2)  # Input of output layer
    a2 = sigmoid(z2)  # Output of output layer

    # Error in output layer
    d2 = (a2 - y)
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(), np.where(z1 > 0, 1, 0))

    # Gradient for w1 and w2
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)

    # Updating parameters
    w1 = w1 - (alpha * (w1_adj))
    w2 = w2 - (alpha * (w2_adj))

    return w1, w2

def train(x, Y, w1, w2, alpha=0.01, epochs=10, batch_size=5):
    acc = []
    losss = []
    for j in range(epochs):
        l = []
        for i in range(0, len(x), batch_size):
            batch_x = np.array(x[i:i + batch_size])
            batch_y = np.array(Y[i:i + batch_size])

            for k in range(len(batch_x)):
                out = f_forward(batch_x[k], w1, w2)
                l.append(loss(out, batch_y[k]))
                w1, w2 = back_prop(batch_x[k], batch_y[k], w1, w2, alpha)
        
        print(f"Epoch {j + 1}, Accuracy: {(1 - (sum(l) / len(x))) * 100:.2f}%")
        acc.append((1 - (sum(l) / len(x))) * 100)
        losss.append(sum(l) / len(x))
    return acc, losss, w1, w2

def predict(x, w1, w2):
    Out = f_forward(x, w1, w2)
    maxm = 0
    k = 0
    for i in range(len(Out[0])):
        if(maxm<Out[0][i]):
            maxm = Out[0][i]
            k = i
    letter = chr(65 + k)
    print(f"Image is of letter {letter}.")
    
    plt.imshow(x.reshape(5, 6))
    plt.show()   
    
def hyperparameter_tuning(x, Y, epochs_list, alpha_list, batch_size_list):
    best_acc = 0
    best_params = {}
    for epochs in epochs_list:
        for alpha in alpha_list:
            for batch_size in batch_size_list:
                w1 = generate_wt(30, 15)  # Hidden layer with 55 neurons
                w2 = generate_wt(15, 26)  # Output layer with 26 neurons
                acc, losss, w1, w2 = train(x, Y, w1, w2, alpha, epochs, batch_size)
                final_acc = acc[-1]
                if final_acc > best_acc:
                    best_acc = final_acc
                    best_params = {
                        'epochs': epochs,
                        'alpha': alpha,
                        'batch_size': batch_size,
                        'w1': w1,
                        'w2': w2
                    }
    return best_acc, best_params

    
w1 = generate_wt(30, 55) #increased number of hidden layers to 10
w2 = generate_wt(55, 26) #increased number of hidden layers to 10
print("w1: ****\n", w1, "\n\n", "w2: *****\n",w2)


"""The arguments of train function are data set list x, 
correct labels y, weights w1, w2, learning rate = 0.1, 
no of epochs or iteration.The function will return the
matrix of accuracy and loss and also the matrix of 
trained weights w1, w2"""
 
acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 10)


import matplotlib.pyplot as plt1
 
# plotting accuracy
plt1.plot(acc)
plt1.ylabel('Accuracy')
plt1.xlabel("Epochs:")
plt1.show()
 
# plotting Loss
plt1.plot(losss)
plt1.xlabel("Epochs:")
plt1.ylabel('Loss')
plt1.show()

print(w1, "\n", w2)

"""
The predict function will take the following arguments:
1) image matrix
2) w1 trained weights
3) w2 trained weights
"""
epochs_list = [50, 100, 150]
alpha_list = [0.1, 0.01, 0.001]
batch_size_list = [5, 10, 20]

best_acc, best_params = hyperparameter_tuning(x, y, epochs_list, alpha_list, batch_size_list)
print(f"Best Accuracy: {best_acc:.2f}%")
print(f"Best Parameters: {best_params}")

# Use the best parameters for prediction
w1, w2 = best_params['w1'], best_params['w2']
for curr in range(len(x)):
    predict(x[curr], w1, w2)
