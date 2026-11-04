import numpy as np
J=[((0.0388353,-8.97657e-09,0.0624),(3.14159,4.18253e-17,-3.14159)),
((-0.0303992,-0.0182778,-0.0542),(-1.5708,-1.5708,0.0)),
((-0.11257,-0.028,1.73763e-16),(-3.63608e-16,8.74301e-16,1.5708)),
((-0.1349,0.0052,3.62355e-17),(4.02456e-15,8.67362e-16,-1.5708)),
((5.55112e-17,-0.0611,0.0181),(1.5708,0.0486795,3.14159)),
((-0.0079,-0.000218121,-0.0981274),(0.0,3.14159,0.0))]
def rx(a):c,s=np.cos(a),np.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def ry(a):c,s=np.cos(a),np.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rz(a):c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def fk(q,upto=6):
    R=np.eye(3);p=np.zeros(3)
    for i in range(upto):
        t,(r,pi,y)=J[i]
        p=p+R@np.array(t);R=R@rz(y)@ry(pi)@rx(r)
        if i<5:R=R@rz(q[i])
    return p
Z5=[0.0]*5
t1=fk(Z5,1); o2=fk(Z5,2); o3=fk(Z5,3); o4=fk(Z5,4); o5=fk(Z5,5)
b2=o3-o2; b3=o4-o3; g0=o5-o4
LO=np.array([-1.91986,-1.74533,-1.69,-1.65806,-2.74385]); HI=np.array([1.91986,1.74533,1.69,1.65806,2.84121])
q5s=np.linspace(LO[4],HI[4],2001)
tipq=lambda q5: fk([0,0,0,0,q5])-o5
U=np.array([tipq(a) for a in q5s]); ux=U[0,0]*0+np.mean(U[:,0]); uzm=U[:,2].mean(); 
g=g0+np.array([ux,0,uzm])
# bases: (var combo, lo, hi, N)
NB={'q1':13,'t2':9,'t23':12,'t234':26,'q5':4}
combo={'q1':[1,0,0,0,0],'t2':[0,1,0,0,0],'t23':[0,1,1,0,0],'t234':[0,1,1,1,0],'q5':[0,0,0,0,1]}
rng={'q1':(LO[0],HI[0]),'t2':(LO[1],HI[1]),'t23':(LO[1]+LO[2],HI[1]+HI[2]),'t234':(LO[1]+LO[2]+LO[3],HI[1]+HI[2]+HI[3]),'q5':(LO[4],HI[4])}
off={};n=0
for k in NB: off[k]=n; n+=NB[k]
assert n==64
W1=np.zeros((64,5)); B1=np.zeros(64); info={}
for k in NB:
    lo,hi=rng[k];N=NB[k];h=(hi-lo)/(N-1);s=h/2.5
    kn=lo-h/2+h*np.arange(N+1); m=(kn[:-1]+kn[1:])/2
    for i in range(N):
        W1[off[k]+i]=np.array(combo[k])/(2*s); B1[off[k]+i]=-m[i]/(2*s)
    w=np.pi*s; G=(w/np.sinh(w))*(np.sin(h/2)/(h/2))
    info[k]=(kn,N,G)
def basis(k,f):
    kn,N,G=info[k]; F=np.array([f(x) for x in kn]); D=np.diff(F)/G
    coef=np.zeros(64); coef[off[k]:off[k]+N]=D/2
    return F[0]/G+D.sum()/2, coef
def comb(*terms):
    c0=0.0;cf=np.zeros(64)
    for a,b in terms: c0+=a; cf=cf+b
    return c0,cf
sc=lambda v,t:(v[0]*t,v[1]*t)
cq1=basis('q1',np.cos); sq1=basis('q1',np.sin)
def rotterm(b,which):
    # contribution of Ry(theta) b to x (which=0) or z (which=1)
    if which==0: return lambda th: b[0]*np.cos(th)+b[2]*np.sin(th)
    return lambda th: -b[0]*np.sin(th)+b[2]*np.cos(th)
X0=(o2-t1)[0]; Z0=(o2-t1)[2]
dx=comb((X0,np.zeros(64)),basis('t2',rotterm(b2,0)),basis('t23',rotterm(b3,0)),basis('t234',rotterm(g,0)))
dzr=comb((Z0,np.zeros(64)),basis('t2',rotterm(b2,1)),basis('t23',rotterm(b3,1)),basis('t234',rotterm(g,1)))
dy0=(fk([0,0,0,0,0])-t1)[1]
yy=np.array([tipq(a)[1] for a in np.linspace(0,2*np.pi,721)]); yc=0.5*(yy.max()+yy.min())
uy=lambda q5: tipq(q5)[1]-yc
dy=comb((dy0+yc-tipq(0.0)[1],np.zeros(64)),basis('q5',uy))
# centre dz
zc=Z0*0+ (dzr[0]) # constant part at hidden=0.. use midpoint of range instead
zmid=0.5*(0.0+0.0)
KAPPA=0.06; C0=0.6; T2=-2*np.tanh(C0)/np.cosh(C0)**2; EPS=0.1
prods=[(cq1,dx),(sq1,dy),(sq1,dx),(cq1,dy)]
W2=np.zeros((64,64)); B2=np.zeros(64)
for j,(a,b) in enumerate(prods):
    for r,(sa,sb,sg) in enumerate([(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1)]):
        # unit: tanh(C0 + sg*KAPPA*(a + sb*b))
        u=4*j+r
        W2[u]=sg*KAPPA*(a[1]+sb*b[1]); B2[u]=C0+sg*KAPPA*(a[0]+sb*b[0])
# pass-through for dz (constant folded to b3 after estimating centre)
W2[16]=EPS*dzr[1]; 
B2[16]=0.0
coef=1/(4*T2*KAPPA**2)
W3=np.zeros((3,64)); B3=np.zeros(3)
sgn=[1,1,-1,-1]
for j in range(4):
    for r in range(4):
        w=coef*sgn[r]*1.0
        W3[[0,0,1,1][j] if False else 0,0]+=0
# px = c*dx + s*dy ; py = -s*dx + c*dy
tgt={0:[(0,1.0),(1,1.0)],1:[(2,-1.0),(3,1.0)]}
for o,lst in tgt.items():
    for j,sg in lst:
        for r in range(4):
            W3[o,4*j+r]+=sg*coef*[1,1,-1,-1][r]
W3[2,16]=1/EPS
# constants: product units cancel constants (symmetric sums) -> they cancel exactly in (+)-(-)
# dz centre: dz = dzr[0] + sum coef*h ; hidden-sum centre approximated by evaluating at mid pose
def dz_at(x): 
    h=np.tanh(W1@x+B1); return dzr[0]+dzr[1]@h
# B3 for z: t1z + dzr[0] const (b2[16]=0 gives tanh(eps*(coef.h)) ); coef.h has offset -> handled by true offset below
B3[0]=t1[0];B3[1]=t1[1];B3[2]=t1[2]+dzr[0]
b2 B2 = .astype(np.float32)
