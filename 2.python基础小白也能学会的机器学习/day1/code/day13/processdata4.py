import numpy as np

data = np.array([
    [1,1],
    [1,2],
    [1,2]
])



mean = (np.mean(data[:, 1]))
std = np.std(data[:,1])

print((data[:, 1] - mean) / std)
