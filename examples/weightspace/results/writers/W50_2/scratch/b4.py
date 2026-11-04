import numpy as np
from terms import *
LIM_=np.array([1.91986,1.74533,1.69,1.65806,2.74385])
def T(k,q): return terms(q)[k]
t0=T(0,[0]*5); t1=T(1,[0]*5)
def term2(A): return T(2,[0,A,0,0,0])
def term3(A): return T(3,[0,0,A,0,0])
def t5(A,q5): return T(5,[0,0,0,A,q5])
def term45(A): return T(4,[0,0,0,A,0])+t5(A,0)
def Vc(A): return (t5(A,0)-t5(A,np.pi))/2
Vs_y=-0.0079
def uv(f): return f(0.0),f(np.pi/2)
Pyconst=(t1[1]+term2(0.3)[1]+term3(0.3)[1]+term45(0.3)[1])
MARGIN=1
def table(f,L,N,sf):
    h=2*L/(N-2*MARGIN); lo=-L-MARGIN*h
    xs=lo+h*np.arange(N+1); m=0.5*(xs[:-1]+xs[1:])
    sg=h*sf/2; kap=np.pi*sg/np.sinh(np.pi*sg); sinc=np.sin(h/2)/(h/2)
    gv=f(xs)/(kap*sinc); al=gv[1:]-gv[:-1]
    return m,al,h*sf,gv[0]
DEF=dict(c1=9,s1=9,P2=5,P3=7,C4=10,S4=10,c5=5,s5=5,Z2=10,Z3=12)
EPS=0.02
def build(N=DEF,sf=0.9,Lr3=3.0,Lr4=3.9,e_p=0.03):
    w1=np.zeros((64,5));b1=np.zeros(64);w2=np.zeros((64,64));b2=np.zeros(64);w3=np.zeros((3,64));b3=np.zeros(3)
    st=dict(idx=0,k=0)
    d1=np.array([1,0,0,0,0.]);d2=np.array([0,1,0,0,0.]);d3=np.array([0,1,1,0,0.]);d4=np.array([0,1,1,1,0.]);d5=np.array([0,0,0,0,1.])
    F={}
    def tab(name,f,wvec,L):
        m,al,s,g0=table(f,L,N[name],sf); w=np.zeros(64)
        for mk,a in zip(m,al):
            i=st['idx']; w1[i]=wvec/s; b1[i]=-mk/s; w[i]=0.5*a; st['idx']+=1
        F[name]=(w,g0+0.5*al.sum())
    tab('c1',np.cos,d1,LIM_[0]);tab('s1',np.sin,d1,LIM_[0])
    def sinfun(f,comp):
        u,v=uv(f);return lambda A:u[comp]*np.cos(A)+v[comp]*np.sin(A)
    # linear features for ridge z tables
    lin_idx={}
    for nm,dv in (('A2',d2),('A3',d3)):
        w1[st['idx']]=EPS*dv; lin_idx[nm]=st['idx']; st['idx']+=1
    tab('P2',sinfun(term2,0),d2,LIM_[1]);tab('P3',sinfun(term3,0),d3,Lr3)
    tab('C4',np.cos,d4,Lr4);tab('S4',np.sin,d4,Lr4)
    tab('c5',np.cos,d5,LIM_[4]);tab('s5',np.sin,d5,LIM_[4])
    assert st['idx']<=64,st['idx']
    def L(*terms,const=0.0):
        w=np.zeros(64);b=const
        for cf,nm in terms: w+=cf*F[nm][0]; b+=cf*F[nm][1]
        return w,b
    def sinus(fv,comp,sc=1.0):  # list of (coef,name) for component of vector-valued sinusoid in A with tables
        pass
    def comp_terms(f,comp,CN,SN,scale=1.0):
        u,v=uv(f); return [(u[comp]*scale,CN),(v[comp]*scale,SN)]
    px=L((1,'P2'),(1,'P3'),*comp_terms(term45,0,'C4','S4'),const=t1[0])
    pz=L(*comp_terms(term45,2,'C4','S4'),const=t0[2]+t1[2])
    a=L(*comp_terms(Vc,0,'C4','S4',1/0.0079)); bb=L(*comp_terms(Vc,2,'C4','S4',1/0.0079))
    c1=L((1,'c1'));s1=L((1,'s1'));s5=L((1,'s5'))
    g=L((0.5,'c5'),const=-0.5)
    c0=0.5;th=np.tanh(c0);tpp=-2*th*(1-th**2)
    def prod2(u,v,e=e_p):
        ids=[]
        for sg in (1,-1):
            for sgn in (1,-1):
                S=(u[0]+sg*v[0], u[1]+sg*v[1]); i=st['k']; st['k']+=1
                w2[i]=sgn*e*S[0]; b2[i]=c0+sgn*e*S[1]; ids.append(i)
        return ids,1/(4*e*e*tpp)
    def prod3(u,v,w,e=0.04):
        ids=[];cs=[]
        for (a_,b_,c_,wt) in [(1,1,1,1),(1,1,-1,-1),(1,-1,1,-1),(-1,1,1,-1)]:
            S=(a_*u[0]+b_*v[0]+c_*w[0], a_*u[1]+b_*v[1]+c_*w[1]); i=st['k']; st['k']+=1
            w2[i]=e*S[0]; b2[i]=e*S[1]; ids.append(i); cs.append(wt*(-3/e**3)/24)
        return ids,cs
    def addp(out,ids,coef,scale):
        for i,sg in zip(ids,(1,1,-1,-1)): w3[out,i]+=scale*coef*sg
    ids,cf=prod2(c1,px); addp(0,ids,cf,1.0)
    ids,cf=prod2(s1,px); addp(1,ids,cf,-1.0)
    ids,cf=prod2(s1,s5); addp(0,ids,cf,Vs_y)
    ids,cf=prod2(c1,s5); addp(1,ids,cf,Vs_y)
    ids,cs=prod3(c1,g,a)
    for i,c in zip(ids,cs): w3[0,i]+=c*2*0.0079
    ids,cs=prod3(s1,g,a)
    for i,c in zip(ids,cs): w3[1,i]+=-c*2*0.0079
    ids,cf=prod2(g,bb); addp(2,ids,cf,2*0.0079)
    ep=0.02
    for (u,out) in [(c1,1),(s1,0)]:
        i=st['k'];st['k']+=1; w2[i]=ep*u[0]; b2[i]=ep*u[1]; w3[out,i]=Pyconst/ep
    b3[0]=t0[0]
    def t2tab(f,li,L,n):
        m,al,sw,g0=table(f,L,n,sf)
        for mk,a_ in zip(m,al):
            i=st['k'];st['k']+=1
            Hm=np.tanh(EPS*mk); sH=sw*EPS*(1-Hm**2)
            w2[i,li]=1/sH; b2[i]=-Hm/sH; w3[2,i]=0.5*a_; b3[2]+=0.5*a_
        b3[2]+=g0
    zf2=sinfun(term2,2);zf3=sinfun(term3,2)
    t2tab(zf2,lin_idx['A2'],LIM_[1],N['Z2'])
    t2tab(zf3,lin_idx['A3'],Lr3,N['Z3'])
    ez=0.05
    i=st['k'];st['k']+=1; w2[i]=ez*pz[0]; b2[i]=ez*pz[1]; w3[2,i]=1/ez
    global FF; FF=F
    return w1,b1,w2,b2,w3,b3
def run(P,q):
    w1,b1,w2,b2,w3,b3=P
    h=np.tanh(w1@q+b1);h2=np.tanh(w2@h+b2);return w3@h2+b3
def ev(P,n=500,seed=1,q5=None):
    rng=np.random.default_rng(seed);e=[]
    for _ in range(n):
        q=rng.uniform(-LIM_,LIM_)
        if q5 is not None:q[4]=q5
        e.append(run(P,q)-fk(q))
    e=np.array(e);n_=np.linalg.norm(e,axis=1);return round(n_.mean()*1000,2),round(n_.max()*1000,2),np.round(np.abs(e).mean(0)*1000,2)
if __name__=="__main__":
    P=build();print(ev(P),ev(P,q5=0.0))
