import pandas as pd
a=int(input("Enter first element of series"))
b=int(input("Enter second element of series"))
c=int(input("Enter third element of series"))
d=int(input("Enter fourth element of series"))
A=[a,b,c,d]
C=["A","B","C","D"]
B=pd.Series(A, index=C)
print(B)
E=[2,5,7,6]
D=pd.Series(E, index=C)
print(D)
print(A+D)
