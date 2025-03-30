import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

with open('lr_samples.json', 'r') as f:
    data = json.load(f)

x = np.array(data['x']).reshape(-1, 1)
y = np.array(data['y'])

x_graph = np.linspace(0, 1).reshape(-1, 1)

for deg in [1, 2, 5, 9, 15]:

    model = LinearRegression()
    env = PolynomialFeatures(deg)

    x_poly = env.fit_transform(x) # 1 x x**2 x**3 x**4 ...
    model.fit(x_poly, y) # learning
    model_line = model.predict(env.transform(x_graph)) # test or our model's line
    plt.plot(x_graph, model_line, label=f'{deg}-order model')

plt.scatter(x, y, label='Noise samples')
plt.xlim(0, 1)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()