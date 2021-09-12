import numpy
import numpy as np
#Run func n times and avg its results
def func_avg(func,n,*args,**kargs):
    v=[]
    for i in range(n):
        res=func(*args,**kargs)
        v.append(res)
    return np.mean(v)

