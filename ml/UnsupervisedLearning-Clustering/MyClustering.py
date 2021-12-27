# Calculate euclidean distance
from math import sqrt
plot1=[1,3]
plot2=[2,5]
euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )
print(round(euclidean_distance,2))

##################
# Change in basis
# Find the change in basis Matrix, M when you move from
# B1  to B2
##################
import numpy as np
B1 = np.array([[1,3,4,2],
              [3,2,4,1],
              [1,2,3,4],
              [3,2,1,1]])
B2 = np.array([[1,1,2,3],
              [3,3,2,1],
              [1,2,2,3],
              [1,2,2,1]])

B1 = B1.T
B2 = B2.T
# B1 = MB2
# M = B2inv * B1
B2inv = np.linalg.inv(B2)
M = B2inv @ B1
print("="*20)
print("M is")
print(M)

################################
# Another approach; change in basis
##############################
import numpy as np

#Create B1
B1=np.array([[1,3,4,2], [3,2,4,1], [1,2,3,4], [3,2,1,1]])
#take its transpose so that columns are vectors of B1
B1=B1.T
# similarly create B2
B2=np.array([[1,1,2,3],[3,3,2,1],[1,2,2,3],[1,2,2,1]])
#note that we didnt take transpose this time
B2inB1 = np.array([])
for b in B2:
    #the reason we didnt take transpose this time is because we wanted to run it through a loop
    #Matrix M^-1 is obtained by writing basis vectors of B2 in B1 as columns of M
    #this for loop will calculate Basis vectors of B2 in B1

    #Here vectors of B2 are in standard basis (i,j,k,l) so to find its representation in B1, we simply take inverse
    #of matrix B1 and multiply it by the vector.
    temp = (np.linalg.inv(B1) @ np.array(b).T)
    B2inB1 = np.append(B2inB1,temp,axis=0)

# take transpose of the matrix as stated above
B2inB1=B2inB1.reshape(4,4).T

#### Thus our required M will just be the inverse of B2inB1
M = np.linalg.inv(B2inB1).round(2)
M


########################
# Continuing the previous question, lets say you
# have a vector v1 = [2,4,5,2] expressed in the
# basis vectors B1,
# what is V1's representation in B2 ?
##########################

# We already mound M in B2 vector space,
# So for V1's rep in B2, we need to simply multiply V1 with M
V1 = np.array([2,4,5,2])
V1=V1.T
V1inB2 = M @ V1
print(V1inB2)

"""
Change in Basis
Consider a Basis given to you in terms of v1=2i+3j and v2=i-2j as
Bv = {2v1-v2,-2v1+5v2}. Bstd is the standard basis {i,j}. Now
Given a vector Sstd[3 2] in Bstd, how will be the matrix that converts
Sstd to its representation in Bv look like?
"""

import numpy as np
# v1 and v2 in terms of i and j
mat_V=np.array([2,3,1,-2]).reshape(2,2).T
#vectors of basis Bv in terms of v1 and v2
B_V1=np.array([2,-1])
B_V2=np.array([-2,5])
#converting Bv in terms of i and j from v1 and v2
M1=mat_V @ B_V1
M2=mat_V @ B_V2
#writing Bv as a matrix and taking its inverse
M=np.array([M1,M2]).T
M=np.linalg.inv(M)
print(M)

"""
Change of Basis
In the previous question, you found the change of Basis matrix. Now use that to find Sstd's representation in Bv.
"""
Sstd = np.array([3,2])
#M1 = np.linalg.inv(M)
Sstd=Sstd.T
Sstd2Bv = M @ Sstd
print(Sstd2Bv)


############################
# Covariance Matrix
###########################
import numpy as np
import pandas as pd
#a = [[2,2],[3,4],[4,5],[5,7],[9,11]]
a = [[2,1],[3,2],[2,1],[5,1]]
b = ['x','y']
data = pd.DataFrame(a, columns=b)
data

np.cov(data.T)




#
