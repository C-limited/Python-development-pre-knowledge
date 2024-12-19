import numpy as np
data = np.loadtxt("cars.csv",delimiter=",",skiprows=1,usecols=(4,5,1))
np.random.shuffle(data)

testdata = data[0:40]
traindata = data[40:-1]

np.savetxt("test-data.csv",testdata,delimiter=",",fmt="%f")
np.savetxt("train-data.csv",traindata,delimiter=",",fmt="%f")
