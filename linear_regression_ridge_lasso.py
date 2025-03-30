import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

with open('lr_samples.json', 'r') as f:
    data = json.load(f)

x = np.array(data['x'])
y = np.array(data['y'])

x = np.append(x, np.array([0.72, 0.76, 0.78])).reshape(-1, 1)
y = np.append(y, np.array([1.5] * 3))

x_graph = np.linspace(0, 1).reshape(-1, 1)

env = PolynomialFeatures(15)
x_poly = env.fit_transform(x)
x_graph_poly = env.transform(x_graph)

for l in [0.0001, 0.001, 0.01]:

    model1 = Lasso(alpha=l)
    
    model1.fit(x_poly, y)
    model_line1 = model1.predict(x_graph_poly)
    plt.plot(x_graph, model_line1, label=f'λ-{l} lasso model')

    model2 = Ridge(alpha=l)

    model2.fit(x_poly, y)
    model_line2 = model2.predict(x_graph_poly)
    plt.plot(x_graph, model_line2, linestyle='--', label=f'λ-{l} ridge model')

    print(f'model1 λ-{l}:', model1.coef_, model1.intercept_)
    print(f'model2 λ-{l}:', model2.coef_, model2.intercept_)


plt.scatter(x, y, label='Noise samples')
plt.xlim(0, 1)
plt.ylim(-3, 3)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()