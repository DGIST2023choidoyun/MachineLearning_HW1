import numpy as np
import matplotlib.pyplot as plt
import json

x = np.linspace(0, 1)
sample_x = np.linspace(0, 1, 10)
norm = np.sin(2 * np.pi * x)

y_noisy = np.sin(2 * np.pi * sample_x) + np.random.normal(0, 0.3, size=sample_x.shape)

plt.plot(x, norm, label='Original graph')
plt.scatter(sample_x, y_noisy, label='Noise samples')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1)
plt.show()

with open('lr_samples.json', 'w') as f:
    json.dump({ 'x': sample_x.tolist(), 'y': y_noisy.tolist() }, f)
