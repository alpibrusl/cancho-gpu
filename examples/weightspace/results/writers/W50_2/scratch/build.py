import numpy as np, sys
from terms import *
LIM_=np.array([1.91986,1.74533,1.69,1.65806,2.74385])
def T(k,q): return terms(q)[k]
# exact planar terms at q1=0,q5=0
def term2(A): return T(2,[0,A,0,0,0])
def term3(A): return T(3,[0,0,A,0,0])
def term45(A): q=[0,0,0,A,0]; return T(4,q)+T(5,q)
t1=T(1,[0,0,0,0,0]); t0=T(0,[0]*5)
base=t1+ (T(2,[0]*5)*0) # constants handled below
# constant parts: term1 (const), y comps of 4/5 are in term45 tables? treat y via k
K=None
def comp(f,i): return lambda A: f(A)[i]
# Px,Pz tables; Py = t1y + term2y(=0)+term3y + term45y  (check const)
ys=[term2(a)[1]+term3(a)[1]+term45(a)[1] for a in np.linspace(-3,3,7)]
Pyconst=t1[1]+np.mean(ys)
EPS=0.02
S_FRAC=0.5
def table(f,L,N):
    h=2*L/(N-2); lo=-L-h
    xs=lo+h*np.arange(N+1); m=0.5*(xs[:-1]+xs[1:])
    var=0.82*(S_FRAC*h)**2
    g=lambda x:f(x)  # variance correction below
    # correction g = f - var/2 f''
    d=1e-3
    gf=lambda x: f(x)-0.5*var*(f(x+d)-2*f(x)+f(x-d))/d**2 if CORR else f(x)
    gv=np.array([gf(x) for x in xs]); al=gv[1:]-gv[:-1]
    return m,al,gv[0]+0.5*al.sum()*0, h*S_FRAC, gv[0]
CORR=True
def build(N1c=10,N1s=10,N2=8,N3=14,N4=19,M2=9,M3=16,M4=29,Lr3=3.0,Lr4=3.9,Lz3=3.0,Lz4=3.9):
    w1=np.zeros((64,5));b1=np.zeros(64);w2=np.zeros((64,64));b2=np.zeros(64);w3=np.zeros((3,64));b3=np.zeros(3)
    idx=0
    # linear features: A2,A3,A4 -> tanh(EPS*A)
    lin=[np.array([0,1,0,0,0.]),np.array([0,1,1,0,0.]),np.array([0,1,1,1,0.])]
    for v in lin: w1[idx]=EPS*v; idx+=1
    # helper to add a table in layer1: returns (indices, coef 0.5*al, const)
    def t1tab(f,wvec,L,N):
        nonlocal idx
        m,al,_,s,g0=table(f,L,N); sc=1/s
        ids=[]
        for mk in m:
            w1[idx]=sc*wvec; b1[idx]=-sc*mk; ids.append(idx); idx+=1
        # value = g0 + sum al*0.5*(1+tanh)
        return np.array(ids),0.5*al,g0+0.5*al.sum()
    ec=np.array([1,0,0,0,0.])
    cid,cco,cc0=t1tab(np.cos,ec,LIM_[0],N1c)
    sid,sco,sc0=t1tab(np.sin,ec,LIM_[0],N1s)
    # Px tables
    px2=t1tab(lambda A:term2(A)[0],lin[0],LIM_[1],N2)
    px3=t1tab(lambda A:term3(A)[0],lin[1],Lr3,N3)
    px4=t1tab(lambda A:term45(A)[0],lin[2],Lr4,N4)
    assert idx<=64,idx
    # u=cos q1 ; v=Px
    Pxc=t1[0]+px2[2]+px3[2]+px4[2]  # constants
    def lincomb(parts,const):
        w=np.zeros(64)
        for ids,co,_ in parts: w[ids]+=co
        return w,const
    wu_c,bu_c=lincomb([(cid,cco,0)],cc0)
    wu_s,bu_s=lincomb([(sid,sco,0)],sc0)
    wv,bv=lincomb([px2,px3,px4],Pxc)
    # layer2: products
    c0=0.5; e=0.03
    th=np.tanh(c0); tpp=-2*th*(1-th**2)
    k=0
    prods=[]
    for (wu,bu) in [(wu_c,bu_c),(wu_s,bu_s)]:
        row=[]
        for sign in (+1,-1):  # S=u+v ; D=u-v
            wS=wu+sign*wv; bS=bu+sign*bv
            for sg in (+1,-1):
                w2[k]=sg*e*wS; b2[k]=c0+sg*e*bS; row.append(k); k+=1
        prods.append(row)
    coef=1/(4*e*e*tpp)
    # x = 0.0388 + c*Px + s*Py ; y = -s*Px + c*Py ; z
    # c*Px
    r=prods[0]; w3[0,r[0]]+=coef;w3[0,r[1]]+=coef;w3[0,r[2]]-=coef;w3[0,r[3]]-=coef
    r=prods[1]; w3[1,r[0]]-=coef;w3[1,r[1]]-=coef;w3[1,r[2]]+=coef;w3[1,r[3]]+=coef
    # passthrough c, s with small gain
    ep=0.02
    for (wu,bu,out_x,out_y) in [(wu_c,bu_c,(0,0.0),(1,Pyconst)),(wu_s,bu_s,(0,Pyconst),None)]:
        pass
    # c passthrough
    w2[k]=ep*wu_c; b2[k]=ep*bu_c; w3[1,k]=Pyconst/ep; kc=k; k+=1
    w2[k]=ep*wu_s; b2[k]=ep*bu_s; w3[0,k]=Pyconst/ep; ks=k; k+=1
    b3[0]=t1[0]*0+0.0388353  # base offset (term0 x); term1.. inside Px already
    b3[2]=t0[2]
    # Z tables in layer2: input h1 lin idx
    def t2tab(f,li,L,N):
        nonlocal k
        m,al,_,s,g0=table(f,L,N)
        # input H=tanh(EPS*A); map A->H, widths: s_H = s*EPS*sech^2
        H=np.tanh(EPS*m)
        for mk,a in zip(m,al):
            sH=s*EPS*(1-np.tanh(EPS*mk)**2)
            w2[k,li]=1/sH; b2[k]=-np.tanh(EPS*mk)/sH
            w3[2,k]=0.5*a; b3[2]+=0.5*a; k+=1
        b3[2]+=g0
    t2tab(lambda A:term2(A)[2],0,LIM_[1],M2)
    t2tab(lambda A:term3(A)[2],1,Lz3,M3)
    t2tab(lambda A:term45(A)[2],2,Lz4,M4)
    b3[2]+=t1[2]
    assert k<=64,k
    return w1,b1,w2,b2,w3,b3
def run(P,q):
    w1,b1,w2,b2,w3,b3=P
    h=np.tanh(w1@q+b1);h2=np.tanh(w2@h+b2);return w3@h2+b3
def err(P,n=300,seed=0):
    rng=np.random.default_rng(seed);e=[]
    for _ in range(n):
        q=rng.uniform(-LIM_,LIM_);e.append(np.linalg.norm(run(P,q)-fk(q)))
    return np.mean(e)*1000, np.max(e)*1000
if __name__=="__main__":
    P=build()
    print(Pyconst,err(P))
