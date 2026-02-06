import pandas as pd
import numpy as np
# Online Python - IDE, Editor, Compiler, Interpreter
arr1=np.array([5,1])
arr2=np.array([5,1])

arr = np.concatenate((arr1,arr2),axis=0)

print(arr)

df=pd.DataFrame(arr)
print(df)

def sum(a, b):
    return (a + b)