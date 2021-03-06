import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

i=1
N=10
for i in range(0,7): 
	sum=0
	xin=[]
	yin=[]
	zin=[]
	xout=[]
	yout=[]
	zout=[]

	data=4*np.random.random_sample((N,3))-2

	for d in data:
		if (m.sqrt(d[0]**2+d[1]**2+d[2]**2)<2.0):
			xin.append(d[0])
			yin.append(d[1])
			zin.append(d[2])
			sum+=1
		else:
			xout.append(d[0])
			yout.append(d[1])
			zout.append(d[2])	
	pi=8*(sum/N)*(3/4)
	print(pi)
	i=i+1
	N=N*10
	fig=plt.figure()
	plot1= fig.add_subplot(111,projection='3d')
	plot1.scatter3D(xin,yin,zin,c='b',marker='x')
	plot1.scatter3D(xout,yout,zout,c='r',marker='x')
	plt.show()
