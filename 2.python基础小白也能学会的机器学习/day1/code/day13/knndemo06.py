import numpy as np
feature = np.array([
    [-121,47,33],
    [-121.2,46.5,333],
    [-122,46.3,32],
    [-120.9,46.7,323],
    [-120.1,46.2,32]
])
label = np.array([
    200,210,250,215,232
])
predictPoint = np.array([-121,46,323])
matrixtemp = (feature - predictPoint)
matrixtemp2 = np.square(matrixtemp)
#axis=1 代表逐行相加
sortindex = np.argsort(np.sqrt(np.sum(matrixtemp2, axis=1)))
sortlabel = label[sortindex]
k = 3
predictprice = np.sum(sortlabel[0:k])/k
print("预测的房价是{}万".format(predictprice))