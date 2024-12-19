import numpy as np

weight = np.array([
[  0.96926278],
 [-19.12845077]
])

feature = np.array(
    [3,1]
)

print(1  /  (1 + np.exp(-np.dot(feature,weight))))