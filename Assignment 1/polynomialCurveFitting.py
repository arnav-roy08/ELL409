import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

df = pd.read_csv('group23.txt', sep = ' ')
df

import numpy
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.isnull().sum()

df_X = df['x']
df_y = df['y']

train=df.sample(frac=0.8,random_state=150) 
test=df.drop(train.index)

X_train = train['x']
X_train = X_train.to_numpy()

X_test = test['x']
X_test = X_test.to_numpy()


y_train = train['y']
y_train = y_train.to_numpy()

y_test = test['y']
y_test = y_test.to_numpy()


X_train_dup = X_train
X_test_dup = X_test
#print(X_train) 
#print(y_train)

def normalize(value):
    return (value - np.mean(value))/(np.max(value) - np.min(value))
def transform(x, degree):
    m = len(x)
    X_trans = np.ones((m,1))
    X_trans = np.append(X_trans, normalize(x).reshape( -1, 1 ), axis = 1)
    for j in range(2,degree+1):
        x_pow = normalize(x**j)
        X_trans = np.append(X_trans, x_pow.reshape( -1, 1 ), axis = 1)
    return X_trans

###################   Different error functions and their derivative wrt w   ###################
def mean_squared_error(h, y_train , X_train):
    cost = (1/(2*m))*np.sum((y_train - h) ** 2)
    cost = np.squeeze(cost)
    return cost
def grad_mean_squared_error(h , y_train , X_train):
    dw = (1 / m) * np.dot((h - y_train), X_train)
    return dw



def mean_absolute_error(h , y_train, X_train):
    cost = (1/(2*m))*np.sum(abs(y_train - h))
    cost = np.squeeze(cost)
    return cost
def grad_mean_absolute_error(h , y_train, X_train):
    dw = np.dot(((h - y_train)/ abs(y_train - h)), X_train) 
    return dw




def logcosh(h ,  y_train , X_train):
    cost = (1/(2*m))*np.sum(np.log(np.cosh(y_train - h)))
    cost = np.squeeze(cost)
    if( math.isinf(cost) == True): #as cosh gives very high values for y_train - h > 100 thus we have used mean absolute error just to display the error if the value of cosh is very high. We have actually used the derivative of logcosh always to compute the weights. abs(x) - log(2) for large values 
        cost = (1/(2*m))*np.sum(abs(y_train - h)) 
    return cost
def grad_logcosh(h ,  y_train , X_train):
    dw = np.dot(np.tanh(h -y_train) , X_train)
    return (dw)



def huber(h ,  y_train , X_train):
    delta = 1
    loss = np.where(np.abs(y_train-h) < delta , 0.5*((y_train-h)**2), delta*np.abs(y_train - h) - 0.5*(delta**2))
    return np.sum(loss)


def grad_huber(h ,  y_train , X_train):
    delta = 1
    if(np.sum(np.abs(y_train-h)) < 0.5*delta*m):
        return (1 / m) * np.dot((h - y_train), X_train)
    else:
        return delta * np.dot(((h - y_train)/ abs(y_train - h)), X_train)
    
    

def mean_sqaured_log_error(h, y_train , X_train):
    cost = (1000000/(2*m))*np.sum(( np.log(y_train+10000) - np.log(h+10000)) ** 2)
    cost = np.squeeze(cost)
    return cost

def grad_mean_sqaured_log_error(h, y_train , X_train):
    dw = np.dot((( np.log(y_train+10000) - np.log(h+10000)) / (h+10000)) , X_train)    
    return -(dw*1000000)/m



#def cosine_similarity


####################   parameters used in the model   #################### 

degree = 6
learning_rate = 0.1
iterations = 1000000
lambd = 0.00001
numberOfPointsTrain = 80
numberOfPointsTest = 20
errorFunction = mean_squared_error
gradErrorFunction = grad_mean_squared_error
tolerance = 0.0000001

##########################################################################


m  = y_train.shape[0]
x = numpy.append(X_train, X_test, axis=0)
x  = transform(x, degree)
X_train = x[:numberOfPointsTrain]
X_test = x[(100-numberOfPointsTest):] 

y_train = y_train[:numberOfPointsTrain]
y_test = y_test[(20-numberOfPointsTest):]

w = np.random.randn(X_train.shape[1])

prev_error = 0
errors = []
for i in range(iterations):
    h = np.dot(X_train, w)
    error =  h - y_train
    dw =  gradErrorFunction(h , y_train , X_train) + (lambd/m)*w 
    w = w - learning_rate * dw
    cost = errorFunction(h, y_train, X_train)
    if(abs(prev_error -  np.sum(cost)) < tolerance):
        break
    prev_error = np.sum(cost)
    if(i % 1000 == 0):
        print("sum of errors in " + str(i) + "th iteration is "+  str(np.sum(cost) ))


def root_mean_square_error(expected, predicted):
    MSE = np.square(np.subtract(expected,predicted)).mean()
    rmse = math.sqrt(MSE)
    return(rmse)

def error_variance(expected, predicted):
    EV = np.square(np.subtract(expected,predicted)).mean()
    return(EV)


Y_pred_test = np.dot(X_test, w)
Y_pred = np.dot(X_train, w)


print("parameters used ")
print(" ")

print("degree : " + str(degree))
print("error Function : " + str(gradErrorFunction))
print("Lamda for L2 regularisation : " + str(lambd))
print("Number of training points: " + str(numberOfPointsTrain))
print("Number of testing points: " + str(numberOfPointsTest))


print(" ")
print(" ")


print("The polynomial is: ")
for i in range(len(w)):
    print(str(w[i]) + "*X^"+ str(i) , end =" ")
    
    
print(" ")
print(" ")

print("The Training root mean square error is " + str(root_mean_square_error(y_train,Y_pred)))
print("The Testing root mean square error is " + str(root_mean_square_error(y_test , Y_pred_test)))


print("noise variance: " + str(error_variance(y_test ,Y_pred_test)))

plt.figure()

plt.scatter( df_X, df_y, color = 'green' , s = 15 )
plt.plot( df_X, df_y, color = 'green', label = 'actual')

plt.scatter(X_train_dup[:numberOfPointsTrain] ,Y_pred, color = 'red', s = 20, label = 'predicted')
plt.scatter( X_test_dup[(20-numberOfPointsTest):], Y_pred_test, color = 'orange', s = 20, label = 'predicted')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Degree of the hypothesis polynomial is ' + str(degree))
plt.legend()

plt.show()

