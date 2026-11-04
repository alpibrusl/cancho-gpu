from build import *
rs=np.random.default_rng(1);errs=[]
for _ in range(2000):
    q=rs.uniform(LO,HI)
    x=q
    h1=np.tanh(W1@x+B1);h2=np.tanh(W2@h1+B2);y=W3@h2+B3
    errs.append(np.linalg.norm(y-fk(q)))
errs=np.array(errs);print(errs.mean()*1000,np.median(errs)*1000,errs.max()*1000)
