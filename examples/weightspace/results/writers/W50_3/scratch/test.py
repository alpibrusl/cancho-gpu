import numpy as np
exec(open('common.py').read())
from fk import fk
def net(q):
    h=np.tanh(W['w1']@q+W['b1']);h=np.tanh(W['w2']@h+W['b2']);return W['w3']@h+W['b3']
lim=np.array([1.91986,1.74533,1.69,1.65806]);rng=np.random.default_rng(0)
err=[]
for _ in range(2000):
    q=np.r_[rng.uniform(-lim,lim),rng.uniform(-2.74385,2.84121)]
    err.append(np.linalg.norm(net(q)-fk(q)))
err=np.array(err);print('units',nu,'mean mm',err.mean()*1e3,'max',err.max()*1e3)
