#建立、训练多层神经网络，并完成模型的检验
import pandas as pd

inputfile1='../data/train_neural_network_data.xls' #训练数据
inputfile2='../data/test_neural_network_data.xls' #测试数据
testoutputfile = '../tmp/test_output_data.xlsx' #测试数据模型输出文件
data_train = pd.read_excel(inputfile1) #读入训练数据(由日志标记事件是否为洗浴)
data_test = pd.read_excel(inputfile2) #读入测试数据(由日志标记事件是否为洗浴)
y_train = data_train.iloc[:,4].values #训练样本标签列
x_train = data_train.iloc[:,5:17].values #训练样本特征
y_test = data_test.iloc[:,4].values #测试样本标签列
x_test = data_test.iloc[:,5:17].values #测试样本特征

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout

#编译模型，损失函数为binary_crossentropy，用adam法求解
# 创建一个Sequential模型对象，构建一个顺序的神经网络模型。
model = Sequential([Dense(20, activation='relu', input_shape=(11,)),#添加一个全连接层(Dense层)。该层具有20个神经元，
                                                                        # 激活函数为ReLU(Rectified Linear Unit)，
                                                                        # 并且指定输入形状为(11,)，即输入数据的特征维度为11。
                    BatchNormalization(),#添加一个批量归一化层。该层用于在训练过程中对输入进行归一化，有助于提高模型的稳定性和训练速度。
                    Dropout(0.2),#添加一个Dropout层。该层用于在训练过程中以指定的概率(此处为0.2)随机丢弃部分神经元的输出，以防止过拟合。
                    Dense(10, activation='relu'),#再次添加一个全连接层。该层具有10个神经元，激活函数为ReLU。
                    BatchNormalization(),#再次添加一个批量归一化层。
                    Dropout(0.2),#再次添加一个Dropout层。
                    Dense(1, activation='sigmoid')])#添加最后一层全连接层。该层具有一个神经元，激活函数为Sigmoid。
                                                        # 这是一个二分类问题的输出层，Sigmoid函数可以将输出限制在0到1之间，表示概率。

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 30, batch_size = 1) #训练模型
# model.save_weights('../model/net.model') #保存模型参数
model.save_weights('net.model')

r = pd.DataFrame(model.predict(x_test), columns = [u'预测结果'])
pd.concat([data_test.iloc[:,:5], r], axis = 1).to_excel(testoutputfile)
model.predict(x_test)
