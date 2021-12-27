# Mapping Nonlinear to Linear
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x=np.random.uniform(low=10, high=35, size=(200,)) + np.random.normal(loc=2, scale=0.5, size=200)
y = np.sqrt(400 - (x-20)**2)+ np.random.normal(loc=20, scale=0.5, size=200)
#plt.scatter(x,y,color=['red','green'])
plt.figure(figsize=(8,6))
plt.scatter(x,y,color='r')
M=20     # -------> This is the hyperplane representaion, chanign value M changes the plot become linear or not. 
p = (x-M)**2
q = (y-M)**2
#plt.scatter(p,q,color=['red','green'])
plt.figure(figsize=(8,6))
plt.scatter(p,q,color='g')
plt.show()







################3
