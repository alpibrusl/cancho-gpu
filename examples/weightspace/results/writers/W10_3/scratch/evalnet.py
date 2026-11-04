import numpy as np
exec(open('common.py').read())
from fk import fk as FK
print('NH1',NH1,'NH2',NH2)
def net(q):
    h1=np.tanh(W1.astype(np.float32)@q.astype(np.float32)+B1.astype(np.float32))
    h2=np.tanh(W2.astype(np.float32)@h1+B2.astype(np.float32))
    return W3.astype(np.float32)@h2+B3.astype(np.float32)
rng=np.random.default_rng(5)
lo=np.array([-1.91986,-1.74533,-1.69,-1.65806,-2.74385]);hi=np.array([1.91986,1.74533,1.69,1.65806,2.84121])
errs=[]
for _ in range(500):
    q=rng.uniform(lo,hi);errs.append(np.linalg.norm(net(q)-FK(q)))
errs=np.array(errs)*1000
print('mean mm',errs.mean(),'max',errs.max())
