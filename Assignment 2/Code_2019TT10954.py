import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

X = pd.read_csv("training_images.txt", sep='\t' , header = None)
X
print(X)
y = pd.read_csv("training_labels.txt", header = None)
y
y.columns =['label']

df = pd.concat([X.T, y.T])
df = df.T
print(df)
shuffle_df = df.sample(frac=1)

train_size = int(0.8 * len(df)) #splitting the data into train and test

train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

y_train = train_set["label"]
X_train = train_set.drop(["label"] , axis = 1)
y_test = test_set["label"]
X_test = test_set.drop(["label"] , axis = 1)
print(X_train)

#encoding
y_train  = pd.get_dummies(y_train.astype(str))
y_train = y_train.astype(float)

y_test  = pd.get_dummies(y_test.astype(str))
y_test = y_test.astype(float)


y_train = y_train.T
y_test = y_test.T

X_train = X_train.T
X_test = X_test.T

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#initialising of weight and bias

W1 = np.random.uniform(low = -(math.sqrt(6/ (300 + X_train.shape[0]))), high=(math.sqrt(6/ (300 + X_train.shape[0]))), size=(300, X_train.shape[0]))
b1 = np.zeros(shape =(300, 1))

W2 = np.random.uniform(low= -(math.sqrt(6/ (300 + y_train.shape[0]))), high= (math.sqrt(6/ (300 + y_train.shape[0]))), size=(y_train.shape[0], 300))
b2 = np.zeros(shape =(y_train.shape[0], 1))

print(W1.size)


#activation function
def sigmoid(Z):
  A = 1 / (1 + np.exp(-Z))
  return A

def tanh(Z):
    A = np.tanh(Z)
    return A


def relu(Z):
  A = np.maximum(0, Z)
  return A


def leaky_relu(Z):
  A = np.maximum(0.1 * Z, Z)
  return A


#forward propogation
def forward_prop(X_train, W1, W2, b1, b2):
  A1 = X_train

  Z2 = np.dot(W1, A1) + b1
  A2 = sigmoid(Z2)
  Z3 = np.dot(W2, A2) + b2
  A3 = sigmoid(Z3)

  cache = {"Z2": Z2,
           "A2": A2,
           "Z3": Z3,
           "A3": A3}

  return A3, cache


#cost function
def compute_cost(A3, y_train):
  m = y_train.shape[1]
  cost = - (1 / m) * np.sum(np.multiply(y_train, np.log(A3)) + np.multiply(1 - y_train, np.log(1 - A3)))
  cost = np.squeeze(cost)
  return cost


#gradient of activations
def sigmoid_gradient(A2):
  return (A2 * (1 - A2))


#backpropogation
def back_propagate(W1, b1, W2, b2, cache):
  lambd = 0.001
  A2 = cache['A2']
  A3 = cache['A3']
  m = y_train.shape[1]
  learning_rate = 0.02

  dZ3 = A3 - y_train
  dW2 = (1 / m) * np.dot(dZ3, A2.T) + (lambd/m)*W2

  dZ2 = np.multiply(np.dot(W2.T, dZ3), sigmoid_gradient(A2))
  dW1 = (1 / m) * np.dot(dZ2, X_train.T) + (lambd/m)*W1

  W1 = W1 - learning_rate * dW1
  W2 = W2 - learning_rate * dW2

  return W1, W2, b1, b2


#training
for i in range(0, 5000):

  A3, cache = forward_prop(X_train, W1, W2, b1, b2)

  cost = compute_cost(A3, y_train)

  print("sum of cost is " + str(np.sum(cost)))
  W1, W2, b1, b2 = back_propagate(W1, b1, W2, b2, cache)

  if i % 1000 == 0:
    print((i, cost))

A3, cache = forward_prop(X_train, W1, W2, b1, b2)

A3
ans = []


#training accuracy
for i in range(0, 4000):
  maxx = 0
  temp = 0
  for j in range(0, 10):
    if (A3[j][i] > maxx):
      maxx = A3[j][i]
      temp = j
  ans.append(temp)

to_predict = train_set["label"]
to_predict = to_predict.values.tolist()
print(to_predict)
positive = 0
for i in range(0, 4000):
  # print(ans[i] , to_predict[i])
  if (ans[i] == to_predict[i]):
    positive = positive + 1

print(positive / 4000)



#testing accuracy

A1 = X_test
Z2 = np.dot(W1, A1) + b1
A2 = sigmoid(Z2)
Z3 = np.dot(W2, A2) + b2
A3 = sigmoid(Z3)

ans = []

for i in range(0, 1000):
  maxx = 0
  temp = 0
  for j in range(0, 10):
    if (A3[j][i] > maxx):
      maxx = A3[j][i]
      temp = j
  ans.append(temp)

# print(ans)


to_predict = test_set["label"]
to_predict = to_predict.values.tolist()
print(to_predict)
positive = 0
for i in range(0, 1000):
  # print(ans[i] , to_predict[i])
  if (ans[i] == to_predict[i]):
    positive = positive + 1

print(positive / 1000)
