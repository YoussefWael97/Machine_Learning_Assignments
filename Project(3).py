#!/usr/bin/env python
# coding: utf-8

# In[369]:


import pandas as pd
import numpy as np
import pandas
import numpy
data = pd.read_csv('house_data_complete.csv')
data=data.sample(frac=1,replace=False)
data= round(data)
y=data['price']
data= data.drop(['id','date','price','waterfront','zipcode','lat','long'],1)


# In[370]:


import os
from matplotlib import pyplot
fig = pyplot.figure()
pyplot.plot(data['bedrooms'],y,'ro',ms=10,mec='k')
pyplot.ylabel('price')
pyplot.xlabel('bedrooms')
        
fig = pyplot.figure()   
pyplot.plot(data['floors'],y,'ro',ms=10,mec='k')
pyplot.ylabel('price')
pyplot.xlabel('floors') 


# In[371]:


mu=data.mean()
sigma=data.std()
normData=(data-mu)/sigma


# In[372]:


noOfTrainSamples=int(normData.shape[0]*(0.6))
Traindata=normData[:noOfTrainSamples]
y1=y[:noOfTrainSamples]

noOfTestSamples=int(normData.shape[0]*(0.2))
Xdata=normData[noOfTrainSamples+1:noOfTrainSamples+1+noOfTestSamples]
y2=y[noOfTrainSamples+1:noOfTrainSamples+1+noOfTestSamples]

Testdata=normData[noOfTrainSamples+2+noOfTestSamples:noOfTrainSamples+2+(noOfTestSamples*2)]
y3=y[noOfTrainSamples+1:noOfTrainSamples+1+noOfTestSamples]
print(Traindata)


# In[373]:


Traindata= np.concatenate([np.ones((m,1)),Traindata],axis=1)
Xdata= np.concatenate([np.ones((y2.size,1)),Xdata],axis=1)
Testdata= np.concatenate([np.ones((y3.size,1)),Testdata],axis=1)


# In[374]:


def computeCost(X, y,R,theta):
    
    m = y.shape[0]  # number of training examples
    
    # You need to return the following variables correctly

    
    J = 0
    h=np.dot(X,theta)
    
    J=(sum((h-y)**2))+((sum((theta)**2))*R) 
    J=J/(2*m)
    return J


# In[375]:


def computeCostWithout(X,y,theta):
    
    m = y.shape[0]  # number of training examples
    
    # You need to return the following variables correctly
    J = 0
    h=np.dot(X,theta)
    
    J=(sum((h-y)**2))
    J=J/(2*m)
    return J


# In[376]:


def gradientDescent(X, y, theta, R, alpha, num_iters):
   
    m = y.shape[0]  # number of training examples
    
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        alphabym=alpha/m
        sumofh0x=np.dot(X,theta)
        theta=theta-((alpha/m)*((np.dot(X.T,sumofh0x-y))+(R*theta)))

        # save the cost J in every iteration
        J_history.append(computeCost(X, y, R,theta))
    
    return theta, J_history


# In[377]:


m=y1.size
X=Traindata
theta=np.zeros(Traindata.shape[1])
theta1_array=np.zeros(Traindata.shape[1])
theta2_array=np.zeros(Traindata.shape[1])
theta3_array=np.zeros(Traindata.shape[1])

num_iters=200
Rarray=np.arange(0.0001,0.1,0.01)
for R in Rarray:
    theta1,J_history1=gradientDescent(X,y1 ,theta,R,0.1,num_iters)
    theta2,J_history2=gradientDescent(X**2,y1 ,theta,R,0.001,num_iters)
    theta3,J_history3=gradientDescent(X**3,y1 ,theta,R,0.0000001,num_iters)

    theta1_array=numpy.concatenate([theta1_array,theta1],axis=0,out=None)
    theta2_array=numpy.concatenate([theta2_array,theta2],axis=0,out=None)
    theta3_array=numpy.concatenate([theta3_array,theta3],axis=0,out=None)


# In[378]:


J1_array=[]
J2_array=[]
J3_array=[]
for i in range(Rarray.size):
    N1= theta1_array[i*15:((i+1)*15)]
    J1_array.append(computeCostWithout(Xdata,y2,N1))
    
for i in range(Rarray.size):
    N2= theta2_array[i*15:((i+1)*15)]
    J2_array.append(computeCostWithout(Xdata,y2,N2))
    
for i in range(Rarray.size):
    N3= theta3_array[i*15:((i+1)*15)]
    J3_array.append(computeCostWithout(Xdata,y2,N3))


# In[379]:


i1=J1_array.index(min(J1_array))
i2=J2_array.index(min(J2_array))
i3=J3_array.index(min(J3_array))

FinalTheta1=theta1_array[i1*15:((i1+1)*15)]
FinalTheta2=theta2_array[i2*15:((i2+1)*15)]
FinalTheta3=theta3_array[i3*15:((i3+1)*15)]

FinalR1=Rarray[i1]
FinalR2=Rarray[i2]
FinalR3=Rarray[i3]
print(i1,FinalTheta1,FinalR1)


# In[380]:


# Plot the convergence graph
pyplot.plot(np.arange(len(J_history1)), J_history1, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[381]:


# Plot the convergence graph
pyplot.plot(np.arange(len(J_history2)), J_history2, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[382]:


# Plot the convergence graph
pyplot.plot(np.arange(len(J_history3)), J_history3, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[383]:


Testingerror1=computeCost(Testdata, y3,FinalR1,FinalTheta1)
print(Testingerror1)


# In[384]:


Testingerror2=computeCost(Testdata, y3,FinalR2,FinalTheta2)
print(Testingerror2)


# In[385]:


Testingerror3=computeCost(Testdata, y3,FinalR3,FinalTheta3)
print(Testingerror3)


# In[ ]:




