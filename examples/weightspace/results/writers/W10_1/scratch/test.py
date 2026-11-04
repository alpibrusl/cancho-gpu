import numpy as np
from build2 import *
from fk import fk
rng=np.random.default_rng(1)
lo=np.array([-1.91986,-1.74533,-1.69,-1.65806,-2.74385]);hi=np.array([1.91986,1.74533,1.69,1.65806,2.84121])
Q=rng.uniform(lo,hi,(2000,5))
Wt=build()
Y=run(*Wt,Q); T=np.array([fk(q)[0] for q in Q])
err=np.linalg.norm(Y-T,axis=1)*1e3
print("mean mm",err.mean(),"max",err.max())
Q2=Q.copy();Q2[:,4]=0
Y=run(*Wt,Q2);T=np.array([fk(q)[0] for q in Q2]);e=np.linalg.norm(Y-T,axis=1)*1e3
print("q5=0 mean",e.mean(),e.max())
