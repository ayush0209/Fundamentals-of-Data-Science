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


row_train_rem=[]
Y_train_rem=[]
row_valid=[]
Y_valid=[]

for i in range(len(row_train)):
	if(i>=(0.7*len(row_train))):
		row_valid.append(row_train[i])
		Y_valid.append(Y_train[i])
	else:
		row_train_rem.append(row_train[i])
		Y_train_rem.append(Y_train[i])

training_matrix_rem=np.array(row_train_rem)
validation_matrix=np.array(row_valid)

regularization_coefficient=[0.000001,0.00001,0.0001,0.001,0.01,0.125,0.25,0.5,0.75,1.0,2.0]
L2_loss_list=[]
W_list=[]

sum_by_col=training_matrix_rem.sum(axis=0)
avg_by_col=[]
for i in range(len(sum_by_col)):
	avg_by_col.append(sum_by_col[i]/len(row_train_rem))


for i in range(1,len(row_train_rem[0])):
	sqsum=0
	for j in range(len(row_train_rem)):
		training_matrix_rem[j][i]=training_matrix_rem[j][i]-avg_by_col[i]
		sqsum+=(training_matrix_rem[j][i])**2
	sqsum=(sqsum/len(row_train_rem))**0.5
	for j in range(len(row_train_rem)):
		training_matrix_rem[j][i]/=sqsum


sum_by_col=validation_matrix.sum(axis=0)
avg_by_col=[]
for i in range(len(sum_by_col)):
	avg_by_col.append(sum_by_col[i]/len(row_valid))


for i in range(1,len(row_valid[0])):
	sqsum=0
	for j in range(len(row_valid)):
		validation_matrix[j][i]=validation_matrix[j][i]-avg_by_col[i]
		sqsum+=(validation_matrix[j][i])**2
	sqsum=(sqsum/len(row_valid))**0.5
	for j in range(len(row_valid)):
		validation_matrix[j][i]/=sqsum

for t in range(len(regularization_coefficient)):

	precision=0.0001
	Loss=0
	prev_Loss=1
	learning_rate=0.1

	W=[]
	partial_derivative=[]

	for i in range(len(data.columns)-1):
		W.append(1)
		partial_derivative.append(0)
	
	while abs(prev_Loss-Loss)>precision:
		W=np.array([W])
		W=W.transpose()
		prediction=np.dot(training_matrix_rem,W)
		prediction=list(prediction.flat)


		W=list(W.flat)

		mod_of_W=0
		for g in range(len(W)):
			mod_of_W+=abs(W[g])

		temp2=0
		for j in range(len(row_train_rem)):
			temp2+=(prediction[j]-Y_train_rem[j])**2
		temp2=temp2/len(row_train_rem)
		temp2+=(regularization_coefficient[t]*mod_of_W)

		for i in range(len(partial_derivative)):
			temp=0
			for j in range(len(row_train_rem)):
				temp+=(prediction[j]-Y_train_rem[j])*training_matrix_rem[j][i]
			partial_derivative[i]=((2*temp)/len(row_train_rem))+(regularization_coefficient[t]*(W[i]/abs(W[i])))

		for i in range(len(W)):
			W[i]=W[i]-(learning_rate*partial_derivative[i])

		prev_Loss=Loss
		Loss=temp2
		
	W_list.append(W)
	prediction=np.dot(validation_matrix,W)
	prediction=list(prediction.flat)
	loss2=0
	for j in range(len(Y_valid)):
		loss2+=(prediction[j]-Y_valid[j])**2
	loss2=loss2/len(Y_test)
	
	L2_loss_list.append(loss2)


	min_loss_index=0
for t in range(len(regularization_coefficient)):
	if(L2_loss_list[t]<L2_loss_list[min_loss_index]):
		min_loss_index=t


print(regularization_coefficient[min_loss_index])

W_chosen=W_list[min_loss_index]

test_matrix=np.array(row_test)
sum_by_col=test_matrix.sum(axis=0)
avg_by_col=[]
for i in range(len(sum_by_col)):
	avg_by_col.append(sum_by_col[i]/len(row_test))

for i in range(1,len(row_test[0])):
	sqsum=0
	for j in range(len(row_test)):
		test_matrix[j][i]=test_matrix[j][i]-avg_by_col[i]
		sqsum+=(test_matrix[j][i])**2
	sqsum=(sqsum/len(row_test))**0.5
	for j in range(len(row_test)):
		test_matrix[j][i]/=sqsum

prediction=np.dot(test_matrix,W_chosen)
prediction=list(prediction.flat)
loss2=0
for j in range(len(Y_test)):
	loss2+=(prediction[j]-Y_test[j])**2
loss2=loss2/len(Y_test)
print(loss2)

plt.plot(regularization_coefficient,L2_loss_list)
plt.xlabel("Regularization Coefficient")
plt.xlim(0.00001,2)
plt.ylabel("L1_Loss")
plt.show()