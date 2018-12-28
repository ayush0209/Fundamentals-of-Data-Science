import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

excel_file='data.xlsx'
data=pd.read_excel(excel_file)


X0=[]
for i in range(len(data[data.columns[0]])):
	X0.append(1.00)
data['X0']=X0


columns_list=data.columns.tolist()
columns_list=columns_list[-1:]+columns_list[:-1]
data=data[columns_list]


X=[]
Y=[]
for i in range(len(data.columns)-1):
	X.append(data[data.columns[i]])
Y=data[data.columns[len(data.columns)-1]]

row_list=[]
for i in range(len(X[0])):
	row_temp=[]
	for j in range(len(X)):
		row_temp.append(X[j][i])
	row_list.append(row_temp)

row_test=[]
Y_test=[]
row_train=[]
Y_train=[]
for i in range(len(row_list)):
	if(i>=len(row_list)-2000):
		row_test.append(row_list[i])
		Y_test.append(Y[i])
	else:
		row_train.append(row_list[i])
		Y_train.append(Y[i])

training_matrix=np.array(row_train)


training_matrix_transpose=training_matrix.transpose()

temp=np.dot(training_matrix_transpose,training_matrix)
temp=np.linalg.inv(temp)
temp=np.dot(temp,training_matrix_transpose)

Y_train=np.array([Y_train])

Y_train=Y_train.transpose()

theta=np.dot(temp,Y_train)


test_matrix=np.array(row_test)
Y_predicted=np.dot(test_matrix,theta)

Y_predicted=list(Y_predicted.flat)


Y_diff=[]
for i in range(len(Y_predicted)):
	Y_diff.append(Y_test[i]-Y_predicted[i])
loss=((sum(i**2 for i in Y_diff))/len(Y_test))
print(loss)
