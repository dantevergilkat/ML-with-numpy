import csv
import numpy as np
import pickle



'''def softmax(y):
    return np.exp(y) / np.sum(np.exp(y), axis=0)'''

def softmax(x, n, epoch=1): # n is the number of samples
    """Compute softmax values for each sets of scores in x."""
    p_pred = np.empty((n, D_out), dtype=np.float64)
    for i in range(n):
        e_x = np.exp(x[i] - np.max(x[i]))
        p_pred[i] = e_x / e_x.sum()
            
    return p_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 10, 784, 200, 10 # Change H for 3 submissions

# Get data from csv
with open('.//digit-recognizer//train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    x_list = []
    y_list = []
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            arr1 = np.asarray(row[1:], dtype=np.float64) # pixel
            arr2 = np.asarray(row[0], dtype=int)  # label
            x_list.append(arr1)
            y_list.append(arr2)

    x_list = np.stack(x_list)
    y_list = np.stack(y_list)
    print(x_list.shape)
    print(y_list.shape)
 
m = 42000 # Number of train samples
# Convert label to one hot vector
onehot_y = np.empty((m, 10))
for i in range(m):
    label = np.zeros(10) # One hot encoder for corresponding class
    label[y_list[i]] = 1  # WHY -1? DELETE IT; transform after fit
    onehot_y[i] = label

# Split
X_valid, X_train = x_list[:7000], x_list[7000:]
y_valid, y_train = onehot_y[:7000], onehot_y[7000:]

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

best_loss = np.infty
best_w = []
n_batches = int(np.ceil(m/N))
for epoch in range(500):
    for batch_index in range(n_batches):
        #if batch_index == 0:
            #print('[+] Epoch ', epoch, ', [+] Batch ', batch_index)
        # Get batch
        np.random.seed(epoch*N+batch_index)
        indices = np.random.randint(len(y_train), size=N)
        x = X_train[indices]
        y = y_train[indices]
        #if batch_index == 0:
            #print('y: ', y)

        # Forward 
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)
        p_pred = softmax(y_pred, y_pred.shape[0]) # --Add--
        #if batch_index == 0:
            #print('p_pred: ', p_pred)

        # Backprop
        grad_y_pred = (p_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    

    # Compute loss and get best model
    h = X_valid.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    p_pred = softmax(y_pred, y_pred.shape[0], 0) # --Add--
    

    # Calculate loss
    eps = 1e-2
    M = np.multiply(y_valid, np.log(p_pred+eps)) # plus epsilon here for not having NaN error
    
    M = np.sum(M, axis=1)
    loss = -np.average(M)
    #loss = -y_valid.T.dot(np.log(p_pred))
    print('[+] Epoch: ', epoch, ', loss=', loss)
    if loss < best_loss:
        best_loss = loss
        best_w = [w1, w2]

print('Best loss:', best_loss)

#w = [w1, w2]
f = open('weight.pickle', 'wb')
f.write(pickle.dumps(best_w))
f.close
