import numpy as np
from design import *
def build(NS=(10,8,14,24,8), delta=0.6, wf=2.0, GP=0.07, GT=0.08, G3=0.12, L3=4.2):
    wv0=F[4][:3,3]+R4@Rz(0.0)@t5
    U=[t2,R2@t3,wv0]
    def comp(k,idx): return lambda s: np.array([(R1@Rz(x)@U[k])[idx] for x in np.atleast_1d(s)])
    dirs={'q1':([1,0,0,0,0],LIM[0],NS[0],wf),'q2':([0,1,0,0,0],LIM[1],NS[1],wf),
          's2':([0,1,1,0,0],LIM[1]+LIM[2],NS[2],wf),'s3':([0,1,1,1,0],L3,NS[3],wf),
          'q5':([0,0,0,0,1],2.8,NS[4],wf)}
    off={};n=0
    for k,v in dirs.items(): off[k]=n;n+=v[2]
    assert n<=64
    W1=np.zeros((64,5));b1=np.zeros(64)
    for k,(d,L,N,w_) in dirs.items():
        x,m,w,M=make(L,N,w_)
        for i in range(N): W1[off[k]+i]=np.array(d)/w;b1[off[k]+i]=-m[i]/w
    def lin(k,f):
        d,L,N,w_=dirs[k];c=coef(L,N,w_,f);v=np.zeros(64);v[off[k]:off[k]+N]=c[:N];return v,c[N]
    def add_(*Ls):
        v=np.zeros(64);c=0.
        for (vv,cc) in Ls: v=v+vv;c+=cc
        return v,c
    Xl=add_(*[lin(k,comp(i,0)) for k,i in (('q2',0),('s2',1),('s3',2))])
    Zl=add_(*[lin(k,comp(i,2)) for k,i in (('q2',0),('s2',1),('s3',2))])
    Yc=t1[1]+sum((R1@U[k])[1] for k in range(3))
    Cl=lin('q1',np.cos);Sl=lin('q1',np.sin)
    wq=lambda q5:F[4][:3,3]+R4@Rz(q5)@t5
    dx=lin('q5',lambda s:np.array([wq(a)[0]-wv0[0] for a in np.atleast_1d(s)]))
    dz=lin('q5',lambda s:np.array([wq(a)[2]-wv0[2] for a in np.atleast_1d(s)]))
    sinP=lin('s3',np.sin);cosP=lin('s3',np.cos)
    rng=np.random.default_rng(1);Qs=rng.uniform(-1,1,(4000,5))*LIM
    Hs=np.tanh(Qs@W1.T+b1)
    def stats(L):
        v=Hs@L[0]+L[1];return v.mean(),np.abs(v-v.mean()).max()
    mX,rX=stats(Xl);mZ,rZ=stats(Zl)
    cen=lambda L,mu:(L[0],L[1]-mu)
    Xc=cen(Xl,mX);Zc=cen(Zl,mZ)
    W2=[];b2=[]
    def add(v,b): W2.append(v);b2.append(b);return len(W2)-1
    tD=np.tanh(delta);k2=-2*tD/np.cosh(delta)**2
    def sc(L,g): return (g*L[0],g*L[1])
    def pair(A,ra,B,rb):
        ga=GP/ra;gb=GP/rb;a=sc(A,ga);c=sc(B,gb);res=[]
        for sgn in (1,-1):
            z=(a[0]+sgn*c[0],a[1]+sgn*c[1])
            res+=[(add(z[0],delta+z[1]),sgn),(add(-z[0],delta-z[1]),sgn)]
        return [(u,s/(4*k2*ga*gb)) for u,s in res]
    def triple(A,ra,B,rb,C,rc):
        ga,gb,gc=G3/ra,G3/rb,G3/rc;a=sc(A,ga);b=sc(B,gb);c=sc(C,gc);res=[]
        for e2 in (1,-1):
            for e3 in (1,-1):
                z=(a[0]+e2*b[0]+e3*c[0],a[1]+e2*b[1]+e3*c[1]);res.append((add(z[0],z[1]),e2*e3))
        return [(u,-s/(8*ga*gb*gc)*1.0*0.5*2) for u,s in res]  # abc = -S/8
    def passthru(L,r):
        e=GT/r;u1=add(e*L[0],e*L[1]);u2=add(2*e*L[0],2*e*L[1]);return [(u1,8/(6*e)),(u2,-1/(6*e))]
    def rmax(L):
        v=Hs@L[0]+L[1];return np.abs(v).max()
    Xm=mX+t1[0]; Zm=mZ+t1[2]
    W3=np.zeros((3,64));b3=np.zeros(3)
    def acc(o,terms,k):
        for u,c in terms: W3[o,u]+=k*c
    # base pairs
    acc(0,pair(Xc,rX,Cl,1.0),-1);acc(1,pair(Xc,rX,Sl,1.0),1)
    PS=passthru(Sl,1.0);PC=passthru(Cl,1.0)
    acc(0,PS,Yc);acc(0,PC,-Xm);acc(1,PC,Yc);acc(1,PS,Xm)
    acc(2,passthru(Zc,rZ),-1)
    # q5 corrections
    rdx=rmax(dx);rdz=rmax(dz)
    T1=triple(dx,rdx,sinP,1.0,Cl,1.0);T2=triple(dx,rdx,sinP,1.0,Sl,1.0)
    acc(0,T1,-1);acc(1,T2,1)
    acc(0,pair(dz,rdz,Sl,1.0),1);acc(1,pair(dz,rdz,Cl,1.0),1)
    acc(2,pair(dx,rdx,cosP,1.0),-1)
    b3[:]=[t0[0],t0[1],t0[2]-Zm]
    n2=len(W2)
    W2m=np.zeros((64,64));B2=np.zeros(64)
    for i in range(n2): W2m[i]=W2[i];B2[i]=b2[i]
    return dict(w1=W1,b1=b1,w2=W2m,b2=B2,w3=W3,b3=b3,n2=n2)
def run(P,Q):
    h1=np.tanh(Q@P['w1'].T+P['b1']);h2=np.tanh(h1@P['w2'].T+P['b2']);return h2@P['w3'].T+P['b3']
def score(P,n=1500,seed=5):
    rng=np.random.default_rng(seed);Q=rng.uniform(-1,1,(n,5))*LIM
    T=np.array([fk(q) for q in Q]);E=np.linalg.norm(run(P,Q)-T,axis=1)*1000
    return E.mean(),np.median(E),E.max()
if __name__=='__main__':
    P=build();print(P['n2'],score(P))
