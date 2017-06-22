import numpy as np

import csv

print("Reading dataset...")

csv.register_dialect('semicolon', delimiter=';')

fin = open("train.csv", 'rt')
reader = csv.reader(fin, dialect='semicolon')

data = [d for d in reader]
names = data[0]
names = names[1:]
data = data[1:]

fin.close()

X = np.asarray(data, dtype = np.float64)

X = np.delete(X, 0, 1)

Y = X[:, [11]]
X = np.delete(X, 11, 1)

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

for x, y in zip(X, Y):
    good = True
    for i in range(7):
        if abs(x[i] - mean[i]) > std[i]:
            good = False
            break
    if x[4] < x[5]:
        good = False
    if good:
        good_data.append(x)
        good_targets.append(y)

print(len(good_data))

X = np.asarray(good_data, dtype = np.float64)

X = (X - X.min(0)) / X.ptp(0)

fout = open("cleaned_train.csv", 'w')

for x, y in zip(X, good_targets):
    for n in x:
        fout.write(str(n))
        fout.write(", ")
    fout.write(str(y[0]))
    fout.write('\n')

fout.close()

