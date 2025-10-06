All the below codes have been tested today on Google Colab.
```
https://share.google/qAqU621grScG0OFpR
```
# NUMPY BASIC CODES

```

# Import Numpy Library
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from IPython.display import Image



list1 = [10,20,30,40,50,60]
list1


# Display the type of an object
type(list1)


# Convert list to Numpy Array
arr1 = np.array(list1)
arr1


# Memory address of an array object
arr1.data



# Display type of an object
type(arr1)


# Datatype of array
arr1.dtype


# Convert Integer Array to FLOAT
arr1.astype(float)


# Generate evenly spaced numbers (space =1) between 0 to 10
np.arange(0,10)


# Generate numbers between 0 to 100 with a space of 10
np.arange(0,100,10)



# Generate numbers between 10 to 100 with a space of 10 in descending order
np.arange(100, 10, -10)



# Shape of Array
arr3 = np.arange(0,10)
arr3.shape



# Size of array
arr3.size


# Bytes consumed by one element of an array object
arr3.itemsize


# Bytes consumed by an array object
arr3.nbytes


# Length of array
len(arr3)


# Generate an array of zeros
np.zeros(10)


# Generate an array of ones with given shape
np.ones(10)



# Repeat 10 five times in an array
np.repeat(10,5)



# Repeat each element in array 'a' thrice
a= np.array([10,20,30])
np.repeat(a,3)


# Generate array of Odd numbers
ar1 = np.arange(1,20)
ar1[ar1%2 ==1]


# Generate array of even numbers
ar1 = np.arange(1,20)
ar1[ar1%2 == 0]



# Create an array of random values
np.random.random(4)


# Generate an array of Random Integer numbers
np.random.randint(0,500,5)



arr2 = np.arange(1,20)
arr2


# Sum of all elements in an array
arr2.sum()

# Cumulative Sum
np.cumsum(arr2)


# Find Minimum number in an array
arr2.min()


#  Find MAX number in an array
arr2.max()


# Find mean of all numbers in an array
arr2.mean()


# Find median of all numbers present in arr2
np.median(arr2)


# Variance
np.var(arr2)

# Standard deviation
np.std(arr2)


# Calculating percentiles
np.percentile(arr2,70)


A = np.array([[1,2,3,0] , [5,6,7,22] , [10 , 11 , 1 ,13] , [14,15,16,3]])
A


# SUM of all numbers in a 2D array
A.sum()

# MAX number in a 2D array
A.max()

# Minimum
A.min()


# Enumerate for Numpy 2D Arrays
for index, value in np.ndenumerate(A):
    print(index, value)



a = np.array([7,5,3,9,0,2])


# Access first element of the array
a[0]

# Access all elements of Array except first one.
a[1:]


# Fetch 2nd , 3rd & 4th value from the Array
a[1:4]


# Get last element of the array
a[-1]


ar = np.arange(1,20)
ar


# Replace EVEN numbers with ZERO
rep1 = np.where(ar % 2 == 0, 0 , ar)
print(rep1)



# Replace 10 with value 99
rep2 = np.where(ar2 == 10, 99 , ar2)
print(rep2)



p2 = np.arange(0,100,10)
p2

# Replace values at INDEX loc 0,3,5 with 33,55,99
np.put(p2, [0, 3 , 5], [33, 55, 99])
p2




a = np.array([10, np.nan,20,30,60,np.nan,90,np.inf])
a


# Search for missing values and return as a boolean array
np.isnan(a)


# Index of missing values in an array
np.where(np.isnan(a))


# Replace all missing values with 99
a[np.isnan(a)] = 99
a


# Check if array has any NULL value
np.isnan(a).any()



# Stack arrays vertically
a = np.zeros(20).reshape(2,-1)
b = np.repeat(1, 20).reshape(2,-1)
a b

np.vstack([a,b])



# Stack arrays horizontally

np.hstack([a,b])
np.hstack([a1,b1])



# Common items between two arrays

c1 = np.array([10,20,30,40,50,60])
c2 = np.array([12,20,33,40,55,60])
np.intersect1d(c1,c2)



# Remove common elements of C1 & C2 array from C1

np.setdiff1d(c1,c2)
```




# PANDAS BASIC CODES


```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import math



# Create series from Nump Array
v = np.array([1,2,3,4,5,6,7])
s1 = pd.Series(v)
s1


# Datatype of Series
s1.dtype


# number of bytes allocated to each item
s1.itemsize



# Number of bytes consumed by Series
s1.nbytes



# Shape of the Series
s1.shape


# number of dimensions
s1.ndim

s1.size

s1.count()


# Create series from List 
s0 = pd.Series([1,2,3],index = ['a','b','c'])
s0


# Modifying index in Series
s1.index = ['a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g']
s1


# Create Series using Random and Range function
v2 = np.random.random(10)
ind2 = np.arange(0,10)
s = pd.Series(v2,ind2)
v2 , ind2 , s


# Creating Series from Dictionary
dict1 = {'a1' :10 , 'a2' :20 , 'a3':30 , 'a4':40}
s3 = pd.Series(dict1)
s3



pd.Series(99, index=[0, 1, 2, 3, 4, 5])


s = [0,1,2,3,4,5,6,7,8,9,10]


# Return all elements of the series
s[:]


# First three element of the Series
s[0:3]

# Last element of the Series
s[-1:]


# Fetch first 4 elements in a series
s[:4]


s2 = s1.copy()
s2

# Append S2 & S3 Series
s4 = s2.append(s3)
s4


s4 = s4.append(pd.Series({'a4': 7}))
s4


v1 = np.array([10,20,30])
v2 = np.array([1,2,3])
s1 = pd.Series(v1) 
s2 = pd.Series(v2)
s1 , s2


# Addition of two series
s1.add(s2)


# Subtraction of two series
s1.sub(s2)


# Increment all numbers in a series by 9
s1.add(9)


# Multiplication of two series
s1.mul(s2)


# Multiply each element by 1000
s1.multiply(1000)


# Division
s1.divide(s2)


# MAX number in a series
s1.max()


# Min number in a series
s1.min()


# Average
s1.mean()


# Median
s1.median()


# Standard Deviation
s1.std()


# Series comparison
s1.equals(s2)


s5 = pd.Series([1,1,2,2,3,3], index=[0, 1, 2, 3, 4, 5])
s5

s5.value_counts()



# creating a dataframe
df = pd.DataFrame()
df


# Create Dataframe using List
lang = ['Java' , 'Python' , 'C' , 'C++']
df = pd.DataFrame(lang)
df



# Add column in the Dataframe
rating = [1,2,3,4]
df[1] = rating
df


# Create Dataframe from Dictionary

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

df2 = pd.DataFrame(data)
df3 = pd.DataFrame(data, index=['row1', 'row2'], columns=['a', 'b'])
df4 = pd.DataFrame(data, index=['row1', 'row2'], columns=['a', 'b' ,'c'])
df5 = pd.DataFrame(data, index=['row1', 'row2'], columns=['a', 'b' ,'c' , 'd'])




dates = pd.date_range(start='2020-01-20', end='2020-01-26')
dates



dates = pd.date_range(start='2020-01-20', periods=7)
dates


M = np.random.random((7,7))
M


dframe = pd.DataFrame(M , index=dates)
dframe



# List Index
dframe.index


# List Column Names
dframe.columns


# Datatype of each column
dframe.dtypes




col1 = np.linspace(10, 100, 30)
col2 = np.random.randint(10,100,30)
df10 = pd.DataFrame({"C1" : col1 , "C2" :col2})
df10

# Delete rows with index values 17,18,19

df10 = df10.drop([17,18,19], axis=0)
df10
```





# LINEAR ALGEBRA BASIC CODES


```

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



v = [3,4]
u = [1,2,3]


w = np.array([9,5,7])


w.shape[0]


a = np.array([7,5,3,9,0,2])

# accessing the first element
a[0]

# accessing complete elements
a[1:]

#accessing last element
a[-1]


# Plotting a vector

v = [3,4]
u = [1,2,3]
plt.plot (v)


# Plotting 2D vector

plt.plot([0,v[0]] , [0,v[1]])
plt.plot([8,-8] , [0,0] , 'k--')
plt.plot([0,0] , [8,-8] , 'k--')
plt.grid()
plt.axis((-8, 8, -8, 8))
plt.show()


# Plotting 3D vector

fig = plt.figure()
ax = Axes3D(fig)
ax.plot([0,u[0]],[0,u[1]],[0,u[2]])
plt.axis('equal')
ax.plot([0, 0],[0, 0],[-5, 5],'k--')
ax.plot([0, 0],[-5, 5],[0, 0],'k--')
ax.plot([-5, 5],[0, 0],[0, 0],'k--')
plt.show()


#Vector Addition

v1 = np.array([1,2])
v2 = np.array([3,4])
v3 = v1+v2
v3 = np.add(v1,v2)
print('V3 =' ,v3)
plt.plot([0,v1[0]] , [0,v1[1]] , 'r' , label = 'v1')
plt.plot([0,v2[0]] , [0,v2[1]], 'b' , label = 'v2')
plt.plot([0,v3[0]] , [0,v3[1]] , 'g' , label = 'v3')
plt.plot([8,-8] , [0,0] , 'k--')
plt.plot([0,0] , [8,-8] , 'k--')
plt.grid()
plt.axis((-8, 8, -8, 8))
plt.legend()
plt.show()



#Scalar Multiplication


u1 = np.array([3,4])
a = .5
u2 = u1*a
plt.plot([0,u1[0]] , [0,u1[1]] , 'r' , label = 'v1')
plt.plot([0,u2[0]] , [0,u2[1]], 'b--' , label = 'v2')
plt.plot([8,-8] , [0,0] , 'k--')
plt.plot([0,0] , [8,-8] , 'k--')
plt.grid()
plt.axis((-8, 8, -8, 8))
plt.legend()
plt.show()

a1 = [5 , 6 ,8]
a2 = [4, 7 , 9]
print(np.multiply(a1,a2))



#Dot Product

a1 = np.array([1,2,3])
a2 = np.array([4,5,6])

dotp = a1@a2
print(" Dot product - ",dotp)



#Cross Product

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
print("\nVector Cross Product ==>  \n", np.cross(v1,v2))


# Angle between vectors

v1 = np.array([4,3])
v2 = np.array([-3,4])
lengthV1 = np.sqrt(np.dot(v1,v1)) 
lengthV2  = np.sqrt(np.dot(v2,v2))
ang = np.rad2deg(np.arccos( np.dot(v1,v2) / (lengthV1 * lengthV2)))
print('Angle between Vectors - %s' %ang)



#Length Of Vector 

v3 = np.array([1,2,3,4,5,6])
length = np.sqrt(np.dot(v3,v3))
length

v3 = np.array([1,2,3,4,5,6])
length = np.sqrt(sum(np.multiply(v3,v3)))
length




# Normalized Vector


v1 = [2,3]
length_v1 = np.sqrt(np.dot(v1,v1))
norm_v1 = v1/length_v1
length_v1 , norm_v1




# Inner and Outer Product

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
np.inner(v1,v2)

print("\n Inner Product ==>  \n", np.inner(v1,v2))
print("\n Outer Product ==>  \n", np.outer(v1,v2))




# Matrix Creation

A = np.array([[1,2,3,4] , [5,6,7,8] , [10 , 11 , 12 ,13] , [14,15,16,17]])
A

type(A)

A.shape


#Zero Matrix


np.zeros(9).reshape(3,3)

np.zeros((3,3))



# Matrix of Ones

np.ones(9).reshape(3,3)

np.ones((3, 3))



# Matrix with Random Number

X = np.random.random((3,3))
X



# Identity Matrix


I = np.eye(9)
I



# Diagonal Matrix :-

D = np.diag([1,2,3,4,5,6,7,8])
D



# Matrix Addition

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

C = np.add(M,N,dtype = np.float64)
print("\n Matrix Addition using np.add  ==>  \n", C)



# Matrix Subtraction


C = np.subtract(M,N,dtype = np.float64)
print("\n Matrix Subtraction using np.subtract  ==>  \n"




# Matrix Scalar Multiplication


M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

C = 10

print("\nMatrices Scalar Multiplication ==>  \n", np.multiply(C,M))



# Matrix Transpose


M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\nTranspose of M ==>  \n", M.T)



# Matrix Determinant

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nDeterminant of M ==>  ", np.linalg.det(M))




# Matrix Rank

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nRank of M ==> ", np.linalg.matrix_rank(M))




# Inverse of Matrix


M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nInverse of M ==> \n", np.linalg.inv(M))




# Trace of Matrix

M = np.array([[1,2,3],[4,-3,6],[7,8,0]])

print("\n Matrix (M)  ==>  \n", M)

print("\nTrace of M ==> ", np.trace(M))



# Matrix Dot Product
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)

print("\n Matrix Dot Product ==> \n", M@N)




# Matrix Division
M = np.array([[1,2,3],[4,-3,6],[7,8,0]])
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n First Matrix (M)  ==>  \n", M)
print("\n Second Matrix (N)  ==>  \n", N)


print("\n Matrix Division (M/N)   ==> \n", M/N)




# Sum of all elements in Matrix
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)


print ("Sum of all elements in a Matrix  ==>")
print (np.sum(N))



# Column wise addition

N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)

print ("Column-Wise summation ==> ")
print (np.sum(N,axis=0))



# Row Wise addition
N = np.array([[1,1,1],[2,2,2],[3,3,3]])

print("\n Matrix (N)  ==>  \n", N)

print ("Row-Wise summation  ==>")
print (np.sum(N,axis=1))
```
