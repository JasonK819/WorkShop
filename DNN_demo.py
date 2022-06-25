import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import Func
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv(r'D:\codeProject\PycharmProject\WorkShop\Dataset\train.csv', names=['Rotation Rate', 'Feed Rate', 'Product Type', 'Eligibility rate', 'Punctuality'])

X1 = df['Rotation Rate']
X2 = df['Feed Rate']
X3 = df['Product Type']

X1 = Func.Normalization(X1)
X2 = Func.Normalization(X2)
X3 = Func.Normalization(X3)

X_df = [X1, X2, X3]
X = np.array(X_df).T
X = X.tolist()
X = torch.FloatTensor(X)
# print(X)

Y1 = df['Eligibility rate']
Y2 = df['Punctuality']

Y1 = Func.Normalization(Y1)
Y2 = Func.Normalization(Y2)

Y_df = [Y1, Y2]
Y = np.array(Y_df).T
Y = Y.tolist()
Y = torch.FloatTensor(Y)

# print(Y)


# 定义样本数，输入层维度，隐藏层维度，输出层维度
n, d_input, d_hidden, d_output = 1000, 3, 5, 2

# Create random Tensors to hold inputs and outputs
# X = torch.randn(n, d_input)
# Y = torch.randn(n, d_output)


# 定义网络
class TwoLayerNet(torch.nn.Module):

    def __init__(self, d_input, d_hidden, d_output):
        super(TwoLayerNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_hidden, bias=False),  # 第一层的线性模型
            torch.nn.ReLU(),  # 第二层激活函数
            torch.nn.Linear(d_hidden, d_output)  # 第三层线性模型
        )

    def forward(self, x):
        Y_pred = self.net(x)
        return Y_pred


# Construct our model by instantiating the class defined above
model = TwoLayerNet(d_input, d_hidden, d_output)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

loss_array = []

for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    Y_pred = model(X)

    # Compute and print loss
    loss = criterion(Y_pred, Y)
    print(t, loss.item())
    loss_array.append(loss.item())
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

matplotlib.rc("font",family='FangSong')
i = np.linspace(1,500,500)
plt.plot(i,loss_array)
plt.xlabel('循环次数')
plt.ylabel('均方误差')
plt.show()

torch.save({'model':model.state_dict()},'model.pt')

