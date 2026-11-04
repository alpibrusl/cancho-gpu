from build import *
def err0(P,n=400,seed=1,q5=0.0):
    rng=np.random.default_rng(seed);e=[]
    for _ in range(n):
        q=rng.uniform(-LIM_,LIM_);q[4]=q5;e.append(run(P,q)-fk(q))
    e=np.array(e);return np.linalg.norm(e,axis=1).mean()*1000,np.abs(e).mean(0)*1000
P=build();print(err0(P),err0(P,q5=1.0))
