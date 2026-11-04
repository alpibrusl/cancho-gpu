import numpy as np
from design import *
def build(q5ref=0.0, delta=0.6, GA=0.06, GC=0.05, GT=0.08, extra=None):
    wv=F[4][:3,3]+R4@Rz(q5ref)@t5
    U=[t2,R2@t3,wv]
    def comp(k,idx): return lambda s: np.array([(R1@Rz(x)@U[k])[idx] for x in np.atleast_1d(s)])
    dirs={'q1':([1,0,0,0,0],LIM[0],10,2.0),'q2':([0,1,0,0,0],LIM[1],10,2.0),
          's2':([0,1,1,0,0],LIM[1]+LIM[2],16,2.0),'s3':([0,1,1,1,0],4.2,20,2.0)}
    off={};n=0
    for k,v in dirs.items(): off[k]=n;n+=v[2]
    W1=np.zeros((64,5));b1=np.zeros(64)
    for k,(d,L,N,wf) in dirs.items():
        x,m,w,M=make(L,N,wf)
        for i in range(N): W1[off[k]+i]=np.array(d)/w;b1[off[k]+i]=-m[i]/w
    def lin(k,f):
        d,L,N,wf=dirs[k];c=coef(L,N,wf,f);v=np.zeros(64);v[off[k]:off[k]+N]=c[:N];return v,c[N]
    def comb(parts):
        v=np.zeros(64);c=0.
        for (vv,cc) in parts: v=v+vv;c+=cc
        return v,c
    Xl=comb([lin(k,comp(i,0)) for k,i in (('q2',0),('s2',1),('s3',2))])
    Zl=comb([lin(k,comp(i,2)) for k,i in (('q2',0),('s2',1),('s3',2))])
    Yc=t1[1]+sum((R1@U[k])[1] for k in range(3))
    Xc0=t1[0];Zc0=t1[2]
    Cl=lin('q1',np.cos);Sl=lin('q1',np.sin)
    rng=np.random.default_rng(1);Qs=rng.uniform(-1,1,(4000,5))*LIM
    Hs=np.tanh(Qs@W1.T+b1)
    def rngof(L):
        v=Hs@L[0]+L[1];return v.mean(),np.abs(v-v.mean()).max()
    mX,rX=rngof(Xl);mZ,rZ=rngof(Zl);
    # centered linear functionals (value minus mean)
    def cen(L,mu): return (L[0],L[1]-mu)
    Xc=cen(Xl,mX);Zc=cen(Zl,mZ)
    gA=GA/rX; gC=GC  # cos,sin in [-1,1]
    units=[];outs={0:[],1:[],2:[]}  # each unit: (v,b); outs coefficient
    W2=[];b2=[];W3=np.zeros((3,64));b3=np.zeros(3)
    def add(v,b):
        W2.append(v);b2.append(b);return len(W2)-1
    tD=np.tanh(delta);k2=-2*tD/np.cosh(delta)**2
    def product(A,Cc):  # returns list of (unit index, coef) so sum = (A*C) where A,C are centered linear functionals (value = L.H+c)
        a=(gA*A[0],gA*A[1]);c=(gC*Cc[0],gC*Cc[1])
        res=[]
        for sa,sc,sign in ((1,1,1),(1,-1,-1)):
            z=(sa*a[0]+sc*c[0],sa*a[1]+sc*c[1])
            u1=add(z[0],delta+z[1]);u2=add(-z[0],delta-z[1])
            res+= [(u1,sign),(u2,sign)]
        return [(u,s/(4*k2*gA*gC)) for u,s in res]
    def passthru(L,scale):
        e=GT/scale
        u1=add(e*L[0],e*L[1]);u2=add(2*e*L[0],2*e*L[1])
        return [(u1,8/(6*e)),(u2,-1/(6*e))]
    # centered cos/sin: cos has mean ~0 anyway
    PXc=product(Xc,Cl);PXs=product(Xc,Sl)
    PSn=passthru(Sl,1.0);PCs=passthru(Cl,1.0)
    PZ=passthru(Zc,rZ)
    # x = t0x - (X cos - Y sin);  X = Xc + mX + Xc0
    Xm=mX+Xc0
    for u,c in PXc: W3[0,u]+=-c
    for u,c in PSn: W3[0,u]+=Yc*c
    b3[0]=t0[0]-(-0)  # fixed below
    # X cos = (Xcen+Xm) cos : need passthrough of cos for Xm*cos
    for u,c in PCs: W3[0,u]+=-Xm*c
    # y = X sin + Y cos
    for u,c in PXs: W3[1,u]+=c
    for u,c in PCs: W3[1,u]+=Yc*c
    for u,c in PSn: W3[1,u]+=Xm*c
    b3[0]=t0[0];b3[1]=t0[1]
    # z = t0z - Z
    Zm=mZ+Zc0
    for u,c in PZ: W3[2,u]+=-c
    b3[2]=t0[2]-Zm
    n2=len(W2);
    W2m=np.zeros((64,64));B2=np.zeros(64);
    for i in range(n2): W2m[i]=W2[i];B2[i]=b2[i]
    W3p=np.zeros((3,64));W3p[:,:]=W3
    return dict(w1=W1,b1=b1,w2=W2m,b2=B2,w3=W3p[:, :64],b3=b3,n2=n2)
def run(P,q):
    h1=np.tanh(P['w1']@q+P['b1']);h2=np.tanh(P['w2']@h1+P['b2']);return P['w3']@h2+P['b3']
if __name__=='__main__':
    P=build()
    print('units layer2',P['n2'])
    rng=np.random.default_rng(5);Q=rng.uniform(-1,1,(1000,5))*LIM
    E=np.array([np.linalg.norm(run(P,q)-fk(q)) for q in Q]);print('mean mm',E.mean()*1000,'max',E.max()*1000)
    Q[:,4]=0;E=np.array([np.linalg.norm(run(P,q)-fk(q)) for q in Q]);print('q5=0 mean mm',E.mean()*1000,'max',E.max()*1000)
