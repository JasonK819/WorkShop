# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import Func
import DNN_demo
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    df = pd.read_csv(r'D:\codeProject\PycharmProject\WorkShop\Dataset\test.csv',
                     names=['Rotation Rate', 'Feed Rate', 'Product Type', 'Eligibility rate', 'Punctuality'])

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
    netD = DNN_demo.TwoLayerNet(3, 5, 2)
    state_dict = torch.load('model.pt')
    netD.load_state_dict(state_dict['model'])
    input = X
    output = netD(input)
    # print(output)

    result = output.data.numpy()
    # print(result)

    Eli_pre = np.split(result, [1], 1)[0]
    Punc_pre = np.split(result, [1], 1)[1]

    x = np.linspace(0,199,200)

    plt.figure()
    plt.plot(x, Y1, label='Eli')
    plt.plot(x, Eli_pre, label='Eli_pre')
    plt.legend()

    plt.figure()
    plt.plot(x, Y2, label='Punc')
    plt.plot(x, Punc_pre, label='Punc_pre')
    plt.legend()

    plt.show()