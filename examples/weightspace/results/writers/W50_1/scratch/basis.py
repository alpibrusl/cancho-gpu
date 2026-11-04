import numpy as np
def make(L,N,wf):
    x=np.linspace(-L,L,N+1); h=x[1]-x[0]; m=(x[:-1]+x[1:])/2; w=wf*h
    M=np.hstack([np.tanh((x[:,None]-m[None,:])/w),np.ones((N+1,1))])
    return x,m,w,M
def coef(L,N,wf,f):
    x,m,w,M=make(L,N,wf); return np.linalg.solve(M,f(x))
def ev(L,N,wf,c,s):
    x,m,w,M=make(L,N,wf)
    return np.hstack([np.tanh((s[:,None]-m[None,:])/w),np.ones((len(s),1))])@c
