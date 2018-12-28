import numpy as np
import math as m
import matplotlib.pyplot as plt

i=1
N=10
for i in range(0,7): 
	sum=0
	xin=[]
	yin=[]
	xout=[]
	yout=[]
	
	data=4*np.random.random_sample((N,2))-2

	for d in data:
		if (m.sqrt(d[0]**2+d[1]**2)<2.0):
			xin.append(d[0])
			yin.append(d[1])
			sum+=1
		else:
			xout.append(d[0])
			yout.append(d[1])	
	pi=(sum*4)/N
	print(pi)
	i=i+1
	N=N*10
	fig=plt.figure()
	plot1= fig.add_subplot(111)
	plot1.scatter(xin,yin,c='b',marker='x')
	plot1.scatter(xout,yout,c='r',marker='x')
	plt.axis([-2, 2, -2, 2])
	plt.show()
