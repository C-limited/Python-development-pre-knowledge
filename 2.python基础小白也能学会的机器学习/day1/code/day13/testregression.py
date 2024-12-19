import numpy as np

# data = np.array([
#     [80, 200],
#     [95, 230],
#     [104, 245],
#     [112, 274],
#     [125, 259],
#     [135, 262]
# ])
data = np.loadtxt("cars.csv",delimiter=",",skiprows=1,usecols=(4,1))
feature = data[:,0]
label = data[:,-1]
m = 1
b = 1
learningrate = 0.0000002

def gradientdecent():
    bslop = 0
    for index,x in enumerate(feature):
        bslop = bslop + m*x+b-label[index]
    print("bslop={}".format(bslop))

    mslop = 0
    for index,x in enumerate(feature):
        mslop = mslop + x*(m*x+b-label[index])

    print("mslop={}".format(mslop))
    return (mslop,bslop)

def train():
    for i in range (1,10000000):
        mslop , bslop = gradientdecent()
        global m
        m = m - mslop*learningrate
        global b
        b = b - bslop*learningrate

        if(abs(mslop)<0.1 and abs(bslop) <0.1):
            break

if __name__ == '__main__':
    train()
    print("m={},b={}".format(m,b))

