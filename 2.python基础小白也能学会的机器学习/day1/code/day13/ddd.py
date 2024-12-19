import numpy as np

data = np.array([11,22,33,44,55,66,77])
ones = np.ones(shape=(1,7))
print(np.append(data, ones).reshape((2,7)))