import  numpy as np

#模型
def calc(x1,x2):
   return -0.09881976*x1+(-4.73274969)*x2+40.65357893

#评估模型的准确性

if __name__ == '__main__':
    testdata = np.loadtxt("test-data.csv",delimiter=",")
    feature = testdata[:,0:2]
    label = testdata[:,-1]

    totalerror = 0
    for index,item in enumerate(feature):
        predict = calc(item[0],item[1])
        real = label[index]
        errorrate = (real - predict)/real
        totalerror = totalerror + abs(errorrate)

    error = totalerror/len(label)
    print("错误率：{}".format(error))

