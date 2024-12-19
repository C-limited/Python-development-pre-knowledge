import numpy as np

image = (np.loadtxt("test_image.csv", delimiter=",", max_rows=1))
print(image.reshape(28, 28))
