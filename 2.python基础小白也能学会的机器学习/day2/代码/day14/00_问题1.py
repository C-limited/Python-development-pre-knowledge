import numpy as np

data = np.array([6,6,6,6,6,6])


mean = (np.mean(data))
print(mean)
std = np.std(data)

print((data - mean) / std)
