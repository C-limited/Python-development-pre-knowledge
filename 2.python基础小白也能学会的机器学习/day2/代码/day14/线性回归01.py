import numpy as np
data = np.loadtxt("cars.csv",delimiter=",",skiprows=1,usecols=(4,1))
m = 1
b = 1
xarray = data[:,0]
yreal = data[:,-1]
learningrate = 0.00001

def grandentdecent():
    bslop = 0
    for index, x in enumerate(xarray):
        bslop = bslop + m*x+b - yreal[index]
    bslop = bslop*2/len(xarray)
    print("mse对b求导={}".format(bslop))

    mslop = 0
    for index, x in enumerate(xarray):
        mslop = mslop + (m * x + b - yreal[index])*x
    mslop = mslop * 2 / len(xarray)
    print("mse对m求导={}".format(mslop))
    return (bslop,mslop)

def train():
    for i in range(1,10000000):
        bslop, mslop = grandentdecent()
        global m
        m = m - mslop*learningrate
        global b
        b = b - bslop*learningrate
        if (abs(mslop)<0.5 and abs(bslop)<0.5):
            break

    print("m={},b={}".format(m,b))


if __name__ == '__main__':
    train()

