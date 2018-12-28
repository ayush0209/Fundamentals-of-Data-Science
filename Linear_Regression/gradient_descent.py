#This code normalize features by subtracting value of feature from it's mean.
#Uncommenting the commented part of code will scale down features by dividing by standard deviation after subtracting from mean.

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
test_matrix=np.array(row_test)

sum_by_col=training_matrix.sum(axis=0)
avg_by_col=[]
for i in range(len(sum_by_col)):
	avg_by_col.append(sum_by_col[i]/len(row_train))

for i in range(1,len(row_train[0])):
	#sqsum=0
	for j in range(len(row_train)):
		training_matrix[j][i]=training_matrix[j][i]-avg_by_col[i]
		#sqsum+=(training_matrix[j][i])**2
	#sqsum=(sqsum/len(row_train))**0.5
	#for j in range(len(row_train)):
		#training_matrix[j][i]/=sqsum


precision=0.00001
Loss=0
prev_Loss=1
learning_rate=0.003

W=[]
partial_derivative=[]

for i in range(len(data.columns)-1):
		W.append(0)
		partial_derivative.append(0)


while abs(prev_Loss-Loss)>precision:
	W=np.array([W])
	W=W.transpose()
	prediction=np.dot(training_matrix,W)
	prediction=list(prediction.flat)

	temp2=0
	for j in range(len(row_train)):
		temp2+=(prediction[j]-Y_train[j])**2
	temp2=temp2/len(row_train)

	for i in range(len(partial_derivative)):
		temp=0
		for j in range(len(row_train)):
			temp+=(prediction[j]-Y_train[j])*training_matrix[j][i]		
		partial_derivative[i]=(2*temp)/len(row_train)

	W=list(W.flat)
	for i in range(len(W)):
		W[i]=W[i]-(learning_rate*partial_derivative[i])
	prev_Loss=Loss

	Loss=temp2

sum_by_col=test_matrix.sum(axis=0)
avg_by_col=[]
for i in range(len(sum_by_col)):
	avg_by_col.append(sum_by_col[i]/len(row_test))

for i in range(1,len(row_test[0])):
	#sqsum=0
	for j in range(len(row_test)):
		test_matrix[j][i]=test_matrix[j][i]-avg_by_col[i]
		#sqsum+=(test_matrix[j][i])**2
	#sqsum=(sqsum/len(row_test))**0.5
	#for j in range(len(row_test)):
		#test_matrix[j][i]/=sqsum

prediction=np.dot(test_matrix,W)
prediction=list(prediction.flat)
loss2=0
for j in range(len(Y_test)):
	loss2+=(prediction[j]-Y_test[j])**2
loss2=loss2/len(Y_test)
print(loss2)
