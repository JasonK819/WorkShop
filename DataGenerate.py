import numpy as np
import pandas as pd

x1 = np.random.randint(1000, 10000, 1000)
x2 = np.random.randint(40, 90, 1000)
x3 = np.random.randint(1, 10, 1000)

y1 = (-x1-100*x2-1000*x3+6000)*(-1/23000)
y2 = (7*x1+100*x2-1000*x3-1000)*(1/77000)

Data_Array = [x1, x2, x3, y1, y2]
np_Data = np.array(Data_Array).T

np_Data_train = np.split(np_Data, [800,], 0)[0]
np_Data_test = np.split(np_Data, [800,], 0)[1]
# print(np_Data_train)

pd_data_train = pd.DataFrame(np_Data_train, columns=['Rotation Rate', 'Feed Rate', 'Product Type', 'Eligibility rate', 'Punctuality'])
pd_data_test = pd.DataFrame(np_Data_test, columns=['Rotation Rate', 'Feed Rate', 'Product Type', 'Eligibility rate', 'Punctuality'])
# print(pd_data_train)
# print(pd_data_test)
pd_data_train.to_csv(r'D:\codeProject\PycharmProject\WorkShop\Dataset\train.csv', header=0, index=0)
pd_data_test.to_csv(r'D:\codeProject\PycharmProject\WorkShop\Dataset\test.csv', header=0, index=0)



