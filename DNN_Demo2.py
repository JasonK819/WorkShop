import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

df = pd.read_csv(r'D:\codeProject\PycharmProject\WorkShop\Dataset\train.csv', names=['Rotation Rate', 'Feed Rate', 'Product Type', 'Eligibility rate', 'Punctuality'])

X1 = df['Rotation Rate']
X2 = df['Feed Rate']
X3 = df['Product Type']
X_df = [X1, X2, X3]
X = np.array(X_df).T
X = X.tolist()
X = torch.FloatTensor(X)
print(X)

Y1 = df['Eligibility rate']
Y2 = df['Punctuality']

Y_df = [Y1, Y2]
Y = np.array(Y_df).T
Y = Y.tolist()
Y = torch.FloatTensor(Y)

# print(y)
"""
用pytorch框架实现单层的全连接网络
不使用偏置bias
"""
class TorchModel(nn.Module):    #nn.module是torch自带的库
     def __init__(self, input_size, hidden_size, output_size):
         super(TorchModel, self).__init__()
         self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
         #nn.linear是torch的线性层，input_size是输入的维度，hidden_size是这一层的输出的维度
         self.layer2 = nn.Linear(hidden_size, output_size, bias=False)
         #这个线性层可以有很多个

    def forward(self, x):   #开始计算的函数
         hidden = self.layer1(x)     #传入输入第一层
         # print("torch hidden", hidden)
         y_pred = self.layer2(hidden)       #传入输入第二层
         return y_pred
# x = np.array([1, 0, 0])  #网络输入

#torch实验
torch_model = TorchModel(3, 5, 2)  #这三个数分别代表输入，中间，结果层的维度
#print(torch_model.state_dict())        #可以打印出pytorch随机初始化的权重
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
#通过取字典方式将权重取出来并把torch的权重转化为numpy的
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
#print(torch_model_w1, "torch w1 权重")
#这里你会发现随机初始化的权重矩阵是5×3，所以当自定义模型时需要转置，但是在pytorch中会自动转置相乘
#print(torch_model_w2, "torch w2 权重")
# torch_x = torch.FloatTensor([X])
torch_x = X
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred)
