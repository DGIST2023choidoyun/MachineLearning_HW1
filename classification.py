import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

dataset = fetch_openml('mnist_784')
x = dataset["data"]
y = dataset["target"].astype(int)

np.random.seed(123456)
random_index = np.random.permutation(len(x))
x = x.iloc[random_index]
y = y.iloc[random_index]

accuracy = []
cut = int(len(x) * 0.8)

x_train = x[:cut]
y_train = y[:cut]
x_test = x[cut:]
y_test = y[cut:]

ratio_accuracy = []
for factor in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    cut = int(len(x) * factor)

    x_train = x[:cut]
    y_train = y[:cut]
    x_test = x[cut:]
    y_test = y[cut:]

    model = LogisticRegression(multi_class='multinomial')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    ratio_accuracy.append(accuracy_score(y_test, y_pred))

print(ratio_accuracy)

iter_accuracy = []

for iter in [10, 50, 100, 200, 300, 400, 500, 1000]:

    model = LogisticRegression(multi_class='multinomial', max_iter=iter)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    iter_accuracy.append(accuracy_score(y_test, y_pred))

print(iter_accuracy)
