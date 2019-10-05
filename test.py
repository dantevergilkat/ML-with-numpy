import csv
import numpy as np
import pickle

# Get data from csv
with open('.//digit-recognizer//test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    x = []
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            arr = np.asarray(row, dtype=int) # pixel
            x.append(arr)

    x = np.stack(x)

w = pickle.loads(open('.//weight.pickle', "rb").read())
w1 = w[0]
w2 = w[1]

# forward
h = x.dot(w1)
h_relu = np.maximum(h, 0)
y_pred = h_relu.dot(w2)

with open('submission.csv', mode='w') as csv_file:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(1, y_pred.shape[0]+1):
        output = np.argmax(y_pred[i-1]) # output for one test sample
        writer.writerow({'ImageId':i, 'Label':output})