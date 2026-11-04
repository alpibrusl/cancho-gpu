import numpy as np,sys
from fk import fk
from builder import build
LIM=np.array([1.91986,1.74533,1.69,1.65806,2.8])
def run(W,X):
    x=X.astype(np.float32)
    h1=np.tanh(x@W['w1'].T.astype(np.float32)+W['b1'].astype(np.float32))
    h2=np.tanh(h1@W['w2'].T.astype(np.float32)+W['b2'].astype(np.float32))
    return h2@W['w3'].T.astype(np.float32)+W['b3'].astype(np.float32)
def test(W,n=400,seed=1):
    r=np.random.default_rng(seed)
    lo=np.array([-1.91986,-1.74533,-1.69,-1.65806,-2.74385]);hi=np.array([1.91986,1.74533,1.69,1.65806,2.84121])
    X=r.uniform(lo,hi,(n,5))
    Y=np.array([fk(q)[:3,3] for q in X])
    P=run(W,X)
    e=np.linalg.norm(P-Y,axis=1)*1000
    return e.mean(),e.max(),np.abs(P-Y).mean(0)*1000
if __name__=='__main__':
    W=build()
    print(test(W))
