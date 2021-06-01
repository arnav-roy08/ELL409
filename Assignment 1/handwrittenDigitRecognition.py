import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


X = pd.read_csv("training_images.txt", sep='\t' , header = None)
X
X['784'] = 1 #adding the bias term
y = pd.read_csv("training_labels.txt", header = None)
y
y.columns =['label']

df = pd.concat([X.T, y.T])
df = df.T

shuffle_df = df.sample(frac=1)

# Defining size of training set
train_size = int(0.8 * len(df))

# Spliting the dataset
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

y_train = train_set["label"]
X_train = train_set.drop(["label"] , axis = 1)
y_test = test_set["label"]
X_test = test_set.drop(["label"] , axis = 1)


y_train  = pd.get_dummies(y_train.astype(str))
y_train = y_train.astype(float)

y_test  = pd.get_dummies(y_test.astype(str))
y_test = y_test.astype(float)


y_train = y_train.T
y_test = y_test.T

X_train = X_train.T
X_test = X_test.T

#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def compute_cost(A3, y_train):
    m = y_train.shape[0]
    cost = - (1 / m) * np.sum(np.multiply(y_train, np.log(A3)) + np.multiply(1 - y_train, np.log(1 - A3)))
    cost = np.squeeze(cost)
    return cost    

temp_y = y_train.loc["1.0"]
lambd = 0

list_of_w = [[]]
for number in range(10):
    
    temp_y = y_train.loc[str(number)+ ".0"]
    
    w = np.random.randn(X_train.shape[0]) * 0.01
    m  = temp_y.shape[0]
    #print (len(w),m) - > 784, 4000

    learning_rate = 0.02
    #print (len(y_hat)) - > 4000

    

    for i in range(0, 5000):
        y_hat = sigmoid(np.dot(w, X_train))
        dz =  y_hat - temp_y
        dw = (1 / m) * np.dot(dz, X_train.T)
        w = w - learning_rate * dw  + (lambd/4000)*w
        cost = compute_cost(y_hat, temp_y)
    
        #print("sum of cost is "+  str(np.sum(cost) ))
        if i % 1000 == 0:
            print ("sum of costs in " + str(i) + "th iteration is: "  + str(cost))
        if(np.sum(cost) < 0.0001):
            break
    
    list_of_w.append(w)
            
    temp_y = temp_y.values.tolist()
    ans = 0 
    for i in range(0,4000):
        if(y_hat[i] > 0.3):
            y_hat[i] = 1.0
        else:
            y_hat[i] = 0.0
        if(y_hat[i] == temp_y[i]):
            ans = ans + 1 
    print("accuracy of classifying " + str(number) + " is " + str(ans/40) + "%")


list_of_w.pop(0)

temp = sigmoid(np.dot(list_of_w , X_train))
temp = temp.T
ans = pd.DataFrame(temp)

train_pred = []

for i in range(0,4000):
    maxx = 0 
    temp = 0
    for j in range(0,10):
        if(ans[j][i] > maxx):
            maxx = ans[j][i]
            temp = j
    train_pred.append(temp)

to_predict = train_set["label"]
to_predict = to_predict.values.tolist()
#print(to_predict)

print("Misclassified points in training data; predicted by our model vs label")
positive = 0 
for i in range(0,4000):
    
    if(train_pred[i] == to_predict[i]):
        positive = positive + 1 
    else:
        print(train_pred[i] , to_predict[i])
    
print("Training accuracy is " + str(positive/40) + "%")

temp = sigmoid(np.dot(list_of_w , X_test))
temp = temp.T
ans = pd.DataFrame(temp)

test_pred = []

for i in range(0,1000):
    maxx = 0 
    temp = 0
    for j in range(0,10):
        if(ans[j][i] > maxx):
            maxx = ans[j][i]
            temp = j
    test_pred.append(temp)
    


to_predict = test_set["label"]
to_predict = to_predict.values.tolist()
#print(to_predict)



test_set_x = test_set.drop(["label"] , axis = 1)
test_set_x = test_set_x.values.tolist()
from PIL import Image as im
#print(np.shape(test_set_x[3]))

print("Misclassified points in testing data; predicted by our model vs label")

positive = 0 
for i in range(0,1000):
    
    if(test_pred[i] == to_predict[i]):
        positive = positive + 1
    else:
        print(i,test_pred[i] , to_predict[i])
        test_set_x[i] = test_set_x[i][:-1]
        temp = test_set_x[i]
        
        temp = np.array(temp)
        temp = np.multiply(temp, 256)
        temp = temp.reshape(28,28)
        temp = temp.T

        data = im.fromarray(temp)
        data = data.convert("L")

        #data.save(str(i) + '.png')
        
        
    
print("Testing accuracy is " + str(positive/10) + "%")