#引入一个科学图形库matplotlib
import matplotlib.pyplot as plt
import  numpy as np
points = np.array([
    [0,0],
    [0,5],
    [3,5],
    [3,4],
    [1,4],
    [1,3],
    [2,3],
    [2,2],
    [1,2],
    [1,0],
    [0,0]
])
plt.plot(points[:,0],points[:,1])
for i in range (1,6):
    matrix = np.array([0.5*i, 0])

    newpoints = points + matrix
    plt.plot(newpoints[:, 0], newpoints[:, 1])

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
plt.show()