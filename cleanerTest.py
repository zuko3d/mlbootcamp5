import numpy as np

import csv

print("Reading dataset...")

csv.register_dialect('semicolon', delimiter=';')

fin = open("test2.csv", 'rt')
reader = csv.reader(fin, dialect='semicolon')

data = [d for d in reader]
names = data[0]
names = names[1:]
data = data[1:]

fin.close()

X = np.asarray(data, dtype = np.float32)

X = np.delete(X, 0, 1)

cleared = []

mean = []
std = []

for i in range(11):
    mean.append(X[:, [i]].mean())
    std.append(X[:, [i]].std())

std[1] += 1
std[6] = 110

good_data = []
good_targets = []

for x in X:
    for i in range(7):
        if abs(x[i] - mean[i]) > 3 * std[i]:
            x[i] = mean[i]
    if x[4] < x[5]:
        x[4] = x[5]
    
    good_data.append(x)

print(len(good_data))

X = np.asarray(good_data, dtype = np.float32)

X = (X - X.min(0)) / X.ptp(0)

fout = open("test0.5.csv", 'w')

for x in X:
    for n in x:
        fout.write(str(n))
        fout.write(", ")
    fout.write('\n')

fout.close()

