import numpy as np
# --- kinematic constants (from URDF table, closed-form chain) ---
_J=[((0.0388353,-8.97657e-09,0.0624),(3.14159,4.18253e-17,-3.14159)),
((-0.0303992,-0.0182778,-0.0542),(-1.5708,-1.5708,0.0)),
((-0.11257,-0.028,1.73763e-16),(-3.63608e-16,8.74301e-16,1.5708)),
((-0.1349,0.0052,3.62355e-17),(4.02456e-15,8.67362e-16,-1.5708)),
((5.55112e-17,-0.0611,0.0181),(1.5708,0.0486795,3.14159)),
((-0.0079,-0.000218121,-0.0981274),(0.0,3.14159,0.0))]
def _Rx(a):c,s=np.cos(a),np.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _Ry(a):c,s=np.cos(a),np.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def _Rz(a):c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
_o=np.array(_J[0][0])
def _origins(q5):
    R=np.eye(3);p=np.zeros(3);out=[]
    for i,(xyz,rpy) in enumerate(_J):
        p=p+R@np.array(xyz); out.append(p-_o); R=R@_Rz(rpy[2])@_Ry(rpy[1])@_Rx(rpy[0])
        if i==4:R=R@_Rz(q5)
    return out
# pan frame: R0 = diag(-1,1,-1) so pan about -z; local w = Rz(q1)(p-o); here origins are in base coords -> convert
_R0=_Rz(_J[0][1][2])@_Ry(_J[0][1][1])@_Rx(_J[0][1][0])
def _loc(v):return v  # at q1=0 pan rotation is identity so base-frame offsets are the local coords
_O0=_origins(0.0)
j2,j3,j4=_O0[1],_O0[2],_O0[3]
_e0=_origins(0.0)[5];_ep=_origins(np.pi)[5];_eh=_origins(np.pi/2)[5]
_a=(_e0+_ep)/2;_b=(_e0-_ep)/2;_c=_eh-_a
l2=j3-j2;l3=j4-j3;a4=_a-j4
bz=_b[2]; by=_b[1]; cy=_c[1]
wy0=j2[1]+l2[1]+l3[1]+a4[1]
ox,oy,oz=_J[0][0]
# --- layer-1 groups: sums of tanh edges (smoothed staircase of a sinusoid, gain-corrected) ---
K=64
groups=[]  # (dir vector, edges, h, k)
def mkgroup(d,lo,hi,h):
    n=int(np.ceil((hi-lo+4*h)/h))+1
    e=lo-2*h+h*np.arange(n); k=1.0/h
    return dict(d=np.array(d,float),e=e,h=h,k=k)
H=dict(q1=0.7,q2=0.7,q23=0.8,q234=0.9,q5=1.0)
LIM=dict(q1=1.91986,q2=1.74533,q3=1.69,q4=1.65806,q5l=-2.74385,q5h=2.84121)
G={}
G['q1']=mkgroup([1,0,0,0,0],-LIM['q1'],LIM['q1'],H['q1'])
G['q2']=mkgroup([0,1,0,0,0],-LIM['q2'],LIM['q2'],H['q2'])
G['q23']=mkgroup([0,1,1,0,0],-LIM['q2']-LIM['q3'],LIM['q2']+LIM['q3'],H['q23'])
G['q234']=mkgroup([0,1,1,1,0],-LIM['q2']-LIM['q3']-LIM['q4'],LIM['q2']+LIM['q3']+LIM['q4'],H['q234'])
G['q5']=mkgroup([0,0,0,0,1],LIM['q5l'],LIM['q5h'],H['q5'])
off={};tot=0
for nm,g in G.items(): off[nm]=tot; tot+=len(g['e'])
NH1=tot
def feat(nm,f):
    """affine feature over h1 units approximating f(s) (s = group's linear combo): returns (coef[64], const)"""
    g=G[nm];h,k,e=g['h'],g['k'],g['e']
    kap=lambda w:(np.pi*w/(2*k))/np.sinh(np.pi*w/(2*k))
    Gn=np.sinc(1.0/(2*np.pi)*h) * kap(1.0)  # sinc(wh/2) with w=1 (np.sinc(x)=sin(pi x)/(pi x))
    Gn=(np.sin(h/2)/(h/2))*kap(1.0)
    D=np.array([f(x+h/2)-f(x-h/2) for x in e])
    coef=np.zeros(64); coef[off[nm]:off[nm]+len(e)]=D/(2*Gn)
    const=(f(e[0]-h/2)+D.sum()/2)/Gn
    return coef,const
def lin(*terms):
    c=np.zeros(64);k0=0.0
    for s,(cf,kc) in terms:
        c=c+s*cf;k0+=s*kc
    return c,k0
sc=lambda s,t:(s*t[0],s*t[1])
E5={'cos':lambda x:np.cos(x),'sin':lambda x:np.sin(x)}
Rs=0.5
# main in-plane radial and height as sinusoid sums
def sinus(nm,A,B,kind):
    # A*cos(s)+B*sin(s) as one sinusoid function of group variable
    return feat(nm,lambda s:A*np.cos(s)+B*np.sin(s))
def acc(parts):
    c=np.zeros(64);k0=0.0
    for cf,kc in parts:c=c+cf;k0+=kc
    return c,k0
rho=[sinus('q2',l2[0],l2[2],0),sinus('q23',l3[0],l3[2],0),sinus('q234',a4[0],a4[2],0)]
zz=[sinus('q2',l2[2],-l2[0],0),sinus('q23',l3[2],-l3[0],0),sinus('q234',a4[2],-a4[0],0)]
FR=acc(rho+[(np.zeros(64),j2[0])]); FR=(FR[0]/Rs,FR[1]/Rs)
Zs=0.4
FZ=acc(zz); FZ=(FZ[0]/Zs,FZ[1]/Zs)   # constant j2z added in b3 via oz
Fc1=feat('q1',np.cos);Fs1=feat('q1',np.sin)
FC=feat('q5',np.cos);FS=feat('q5',np.sin)
FSt=feat('q234',np.sin);FCt=feat('q234',np.cos)
# --- layer 2 ---
eps2=0.04;c2=0.6
tc=np.tanh(c2);tpp=-2*tc/np.cosh(c2)**2
epsl=0.02;eps3=0.05
units=[]  # (coef, const) of pre-activation; plus out weights
outw=[]   # (3,) weights
def add(u,ow):units.append(u);outw.append(np.array(ow,float))
def lc(*ts):return lin(*ts)
def double(U,V,out):
    # out: 3-vector scale applied to U*V (U,V unit-scale features)
    for sgn,(S) in ((1,lc((1,U),(1,V))),(-1,lc((1,U),(-1,V)))):
        cf,k0=S
        for s2 in (1,-1):
            add((eps2*s2*cf,c2+eps2*s2*k0),np.array(out)*sgn/(4*eps2**2*tpp))
def triple(U,V,W,out):
    for s2 in (1,-1):
        for s3 in (1,-1):
            cf,k0=lc((1,U),(s2,V),(s3,W))
            add((eps3*cf,eps3*k0),-np.array(out)*s2*s3/(8*eps3**3))
def linear(U,out):
    add((epsl*U[0],epsl*U[1]),np.array(out)/epsl)
double(Fc1,FR,[Rs,0,0]); double(Fs1,FR,[0,-Rs,0])
double(Fs1,FC,[by,0,0]); double(Fs1,FS,[cy,0,0])
double(Fc1,FC,[0,by,0]);  double(Fc1,FS,[0,cy,0])
triple(Fc1,FC,FSt,[bz,0,0]); triple(Fs1,FC,FSt,[0,-bz,0])
double(FC,FCt,[0,0,bz])
linear(Fs1,[wy0,0,0]); linear(Fc1,[0,wy0,0]); linear(FZ,[0,0,Zs])
NH2=len(units)
W1=np.zeros((64,5));B1=np.zeros(64)
for nm,g in G.items():
    for j,ej in enumerate(g['e']):
        W1[off[nm]+j]=g['k']*g['d'];B1[off[nm]+j]=-g['k']*ej
W2=np.zeros((64,64));B2=np.zeros(64);W3=np.zeros((3,64))
for i,(u,ow) in enumerate(zip(units,outw)):
    W2[i]=u[0];B2[i]=u[1];W3[:,i]=ow
B3=np.array([ox,oy,oz+j2[2]*0])  # oz plus constant z of j2 handled below
# constant z: w_z = j2z + sum parts ; FZ already holds parts. add j2z
B3=np.array([ox,oy,oz+j2[2]])
tensor = W2
