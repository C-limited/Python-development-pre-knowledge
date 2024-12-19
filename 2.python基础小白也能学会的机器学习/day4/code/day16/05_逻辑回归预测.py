import numpy as np
#看电视的权重
weight1 = np.array([
[-0.9697223 ],
 [19.13764586]
])
#看书的权重
weight2 = np.array([
[   0.02997379],
 [-0.15720643]
])
#广场舞的权重
weight3 = np.array([
[  0.60928238],
 [-30.35856277]
])

feature = np.array(
    [950,1]
)
tv = np.dot(feature,weight1)
book = np.dot(feature,weight2)
dance = np.dot(feature,weight3)
print("看电视")
print(np.exp(tv)/(np.exp(book)+np.exp(tv)+np.exp(dance)))
print("看书")
print(np.exp(book)/(np.exp(book)+np.exp(tv)+np.exp(dance)))
print("广场舞")
print(np.exp(dance)/(np.exp(book)+np.exp(tv)+np.exp(dance)))