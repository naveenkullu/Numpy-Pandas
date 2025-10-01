# NumPy Basics

---

## 1. Installation & Config

```python
pip3 install numpy
```
Write above command in terminal and download it

```python
import numpy as np

# Check version and config details
print(np.__version__)     
np.show_config()
```
**Output:**
```
1.26.4
<library and compiler details>
```

---

## 2. Creating Arrays
```python
# 0D, 1D, 2D, 3D arrays
a = np.array(42)                        
b = np.array([1, 2, 3, 4, 5])           
c = np.array([[1, 2, 3], [4, 5, 6]])    
d = np.array([[[1, 2, 3], [4, 5, 6]], 
              [[1, 2, 3], [4, 5, 6]]])  

print(a.ndim, b.ndim, c.ndim, d.ndim)   # prints dimensions
```
**Output:**
```
0 1 2 3
```

### Special Arrays
```python
print(np.full((2,2), 7))      # 2x2 matrix filled with 7
print(np.zeros((2,3)))        # 2x3 matrix with all 0s
print(np.ones((2,3)))         # 2x3 matrix with all 1s
print(np.arange(1,10,2))      # range with step
print(np.linspace(1,10,5))    # equally spaced numbers
print(np.eye(3))              # identity matrix
```
**Output:**
```
[[7 7]
 [7 7]]
[[0. 0. 0.]
 [0. 0. 0.]]
[[1. 1. 1.]
 [1. 1. 1.]]
[1 3 5 7 9]
[ 1.    3.25  5.5   7.75 10.  ]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

---

## 3. Data Types
```python
arr = np.array([1,2,3,4])
print(arr.dtype)              # int64 (default)

arr = np.array([1,2,3,4], dtype=np.int8)
print(arr.dtype)              # int8

arr = np.array([1,2,3,4], dtype='f')  
print(arr, arr.dtype)         # float array

print(arr.astype(int))        # convert float to int
```
**Output:**
```
int64
int8
[1. 2. 3. 4.] float32
[1 2 3 4]
```

---

## 4. Arithmetic Operations
```python
arr = np.array([1,2,3,4])
print(arr + 3)         # add scalar
print(arr * 2)         # multiply scalar

a = np.array([1,2,3,4])
b = np.array([1,2,3,4])
print(a + b)           # element-wise addition
print(np.subtract(a,b))# element-wise subtraction
```
**Output:**
```
[4 5 6 7]
[2 4 6 8]
[2 4 6 8]
[0 0 0 0]
```

---

## 5. Random Numbers
```python
print(np.random.rand(4))      # #will generate 4 random numbers between 0 to 1
print(np.random.randn(4))     # give random numbers which close to zero (both positive and negative)
print(np.random.ranf(2))      # #will generate 4 random numbers with random float value in interval [0.0,1.0
print(np.random.randint(3,10,4))  # #will generate 4 random numbers between [3,10]
```
**Output (random):**
```
[0.65 0.12 0.78 0.91]
[-0.23  1.07 -0.56  0.44]
[0.34 0.89]
[4 8 6 9]
```

---

## 6. Useful Math Functions
```python
arr = np.array([8,3,9,10])
print(np.min(arr), np.max(arr))   # min & max
print(np.argmin(arr), np.argmax(arr)) # index of min & max
print(np.sqrt(arr))               # square roots
print(np.cumsum(arr))             # cumulative sum
```
**Output:**
```
3 10
1 3
[2.828 1.732 3.    3.162]
[ 8 11 20 30]
```

---

## 7. Shape and Reshape
```python
arr = np.array([[1,2,3],[4,5,6]])
print(arr.shape)           # shape
print(arr.reshape(3,2))    # reshape
```
**Output:**
```
(2, 3)
[[1 2]
 [3 4]
 [5 6]]
```

---

## 8. Indexing & Slicing
```python
arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr[0,1])         # element
print(arr[1,1:4])       # slice row
print(arr[0:2,2])       # slice column
print(arr[0:2,1:4])     # sub-matrix
```
**Output:**
```
2
[7 8 9]
[3 8]
[[2 3 4]
 [7 8 9]]
```

---

## 9. Copy vs View
Copy owns the data and view doesnot own the data. If we change original array or copy it does not affect each other but change in view or original array reflect in both. View is just a shallow copy which points to same memory of original array.
```python
arr = np.array([1,2,3])
copy_arr = arr.copy()   # deep copy
view_arr = arr.view()   # shallow copy
arr[0] = 42
print(copy_arr)         # unaffected
print(view_arr)         # reflects change
```
**Output:**
```
[1 2 3]
[42  2  3]
```

---

## 10. Advanced Indexing
```python
arr = np.array([10,20,30,40,50])
print(arr[[0,2,4]])            # fancy indexing

mat = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])
print(mat[[0,2],[1,2]])         # row,col pairs
print(arr[arr > 25])            # boolean mask
```
**Output:**
```
[10 30 50]
[2 9]
[30 40 50]
```

---

## 11. Ravel & Flatten
```python
x = np.array([[1,2,3],[4,5,6]])
print(x.ravel())       # returns view
print(x.flatten())     # returns copy
```
**Output:**
```
[1 2 3 4 5 6]
[1 2 3 4 5 6]
```

---

## 12. Insert, Delete, Concatenate
```python
arr = np.array([10,20,30])
print(np.insert(arr,1,99))   # insert element
print(np.delete(arr,2))      # delete element
print(np.concatenate(([1,2,3],[4,5,6])))  # merge arrays
```
**Output:**
```
[10 99 20 30]
[10 20]
[1 2 3 4 5 6]
```

---

## 13. Stacking & Splitting
```python
a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.vstack((a,b)))      # vertical stack
print(np.hstack((a,b)))      # horizontal stack

arr = np.array([1,2,3,4,5,6])
print(np.split(arr,3))       # split into parts

mat = np.array([[1,2,3,4],[5,6,7,8]])
print(np.hsplit(mat,2))      # split columns
```
**Output:**
```
[[1 2 3]
 [4 5 6]]
[1 2 3 4 5 6]
[array([1, 2]), array([3, 4]), array([5, 6])]
[array([[1, 2],
       [5, 6]]), array([[3, 4],
       [7, 8]])]
```

---

## 14. Handling NaN & Infinity
```python
arr = np.array([1,2,np.nan,9])
print(np.isnan(arr))                   # detect NaN
print(np.nan_to_num(arr,nan=0))        # replace NaN

arr = np.array([1,2,np.inf,-np.inf])
print(np.isinf(arr))                   # detect inf
print(np.nan_to_num(arr,posinf=40,neginf=-40))  # replace inf
```
**Output:**
```
[False False  True False]
[1. 2. 0. 9.]
[False False  True  True]
[  1.   2.  40. -40.]
```

---

## 15. Transpose & Dimensions
```python
mat = np.array([[1,2,3],[4,5,6]])
print(mat.T)                  # transpose

arr = np.array([1,2,3])
print(np.expand_dims(arr,axis=0))  # add dimension
print(np.squeeze([[1,2,3]]))       # remove dimension
```
**Output:**
```
[[1 4]
 [2 5]
 [3 6]]
[[1 2 3]]
[1 2 3]
```

---

---

## 16. `np.where()` – Conditional Selection
```python
import numpy as np

# Replace even numbers with 0, keep odd numbers as is
x = np.array([1,2,3,4,5])
y = np.where((x % 2 == 0), 0, x)  
print(y)  # Output: [1 0 3 0 5]
```

---

## 17. Sorting an Array
```python
import numpy as np

# Sort array in ascending order
x = np.array([1,3,2,9,4,5])
y = np.sort(x)
print(y)  # Output: [1 2 3 4 5 9]
```

---

## 18. `searchsorted()` – Find Insertion Index
```python
import numpy as np

# Finds index where '2' should be inserted to keep order
x = np.array([5,9,1,2,6,7])
x1 = np.searchsorted(x, 2)  # By default searches from left
print(x1)  # Output: 1

x2 = np.searchsorted(x, 2, side="right")  # Search from right
print(x2)  # Output: 4
```

---

## 19. Filtering Array with Boolean Indexing
```python
import numpy as np

# Use boolean list to filter array
x = np.array([5,9,1,2,6,7])
f = [True, False, True, True, False, True]  
var3 = x[f]  
print(var3)  # Output: [5 1 2 7]
```

---

## 20. Shuffling Array
```python
import numpy as np

# Randomly shuffle elements of the array (in-place)
x = np.array([5,9,1,2,6,7])
np.random.shuffle(x)
print(x)  # Output: (Random order each time, e.g. [2 7 1 5 6 9])
```

---

## 21. Unique Elements
```python
import numpy as np

# Get unique values
x = np.array([5,9,1,2,6,7,7,8,8,8,9,9])
y = np.unique(x)
print(y)  # Output: [1 2 5 6 7 8 9]

# Unique values with their first index positions
z = np.unique(x, return_index=True)
print(z)

# Unique values with their index positions and frequency count
q = np.unique(x, return_index=True, return_count=True)
print(q)
```

---

## 22. Resize Array
```python
import numpy as np

# Reshape array into (3x2) dimension
x = np.array([5,9,1,2,6,7,7,8,8,8,9,9])
y = np.resize(x, (3,2))
print(y)
# Output:
# [[5 9]
#  [1 2]
#  [6 7]]
```

---

## 23. Defining a Matrix
```python
import numpy as np

# Create a 2D matrix
var1 = np.matrix([[1,2,3], [4,5,6], [7,8,9]])
print(var1)
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
```

---

## 24. Transpose & Swap Axes
```python
import numpy as np

# Transpose a matrix
var1 = np.matrix([[1,2,3], [4,5,6], [7,8,9]])
print(var1.transpose())  
print(var1.T)  # Shortcut for transpose

# Swap rows and columns (same as transpose here)
swapped = np.swapaxes(var1, 0, 1)
print(swapped)
```

---

## 25. Swap Axes in 3D Array
```python
import numpy as np

# Swap axis 0 and 2 in a 3D array
var1 = np.array([[[1,2,3], [4,5,6], [7,8,9]]])
swapped = np.swapaxes(var1, 0, 2)
print(swapped)
# Output:
# [[[1]
#   [4]
#   [7]]
#
#  [[2]
#   [5]
#   [8]]
#
#  [[3]
#   [6]
#   [9]]]
```

---

## 26. Dot Product of Matrices
```python
import numpy as np

# Matrix multiplication (dot product)
var1 = np.matrix([[1,2,3], [4,5,6], [7,8,9]])
var2 = np.matrix([[10,11,12], [13,14,15], [16,17,18]])

var3 = var1 * var2  # Using operator
var4 = np.dot(var1, var2)  # Using np.dot()
print(var3)
print(var4)
# Output:
# [[ 84  90  96]
#  [201 216 231]
#  [318 342 366]]
```

---

## 27. Inverse of a Matrix
```python
import numpy as np

# Find inverse of a matrix
var1 = np.matrix([[1,2,4], [4,5,6], [7,8,9]])
print(np.linalg.inv(var1))  
print(np.linalg.matrix_power(var1, -1))  # Alternative way
# Output:
# [[-1.5         3.         -1.5       ]
#  [ 1.33333333 -2.66666667  1.33333333]
#  [-0.16666667  0.33333333 -0.16666667]]
```

---

## 28. Matrix Power
```python
import numpy as np

# Raise matrix to power 2
var1 = np.matrix([[1,2,4], [4,5,6], [7,8,9]])
print(np.linalg.matrix_power(var1, 2))
# Output:
# [[ 37  48  58]
#  [ 72  99 126]
#  [111 150 189]]
```

---

## 29. Determinant of a Matrix
```python
import numpy as np

# Find determinant of a square matrix
var1 = np.matrix([[1,2,3], [4,5,6], [7,8,9]])
print(var1.shape)       # Output: (3, 3)
print(np.linalg.det(var1))  # Output: 0.0 (since it's singular matrix)
```
---

## 30. To solve eqn.

Let two eqn. be AX = B
2x + y = 9
x + y = 3

```python
import numpy as np

# Make matrix A and B
a = np.array([[2 1] , [1 1]])
b = np.array([ 9 3 ])
x = np.linalg.solve(a , b)
print(x)
---

