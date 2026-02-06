import pandas as pd 
import numpy as np 
import math
df = pd.DataFrame({'X':[10,11,15,20,25,30,31,32,33,34], 'Y':[25,40,70,120,160,170,175,173,178,175]});
print(df)
numTrgt = 200;
x = np.zeros([numTrgt,1])
y = np.zeros([numTrgt,1])
ratio=int(numTrgt/len(df));
print(ratio)

for i in range(numTrgt-ratio):
    c2 = df.iloc[int(math.ceil(i/ratio)-1),0]
    c3 = df.iloc[int(math.ceil(i/ratio)),0]
    x[i]=(c3-c2)*i/ratio+c2;
    if(int(math.ceil(i/ratio)) ==int(math.floor(i/ratio))):
        y[i]=np.interp(x[i],df.iloc[:,0],df.iloc[:,1])
    else:
        y[i]=np.nan
arr=np.concatenate((x,y),axis=1);
df2=pd.DataFrame(arr)
df2.interpolate(method ='linear', axis= 1)
print(df2)
