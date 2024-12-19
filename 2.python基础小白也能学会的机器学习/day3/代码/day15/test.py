import numpy as np

print(np.loadtxt("test_image.csv", delimiter=",", max_rows=1).reshape(28,28))