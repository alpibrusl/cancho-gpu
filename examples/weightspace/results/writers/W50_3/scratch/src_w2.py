import numpy as np
J=[((0.0388353,-8.97657e-09,0.0624),(3.14159,4.18253e-17,-3.14159)),
((-0.0303992,-0.0182778,-0.0542),(-1.5708,-1.5708,0.0)),
((-0.11257,-0.028,1.73763e-16),(-3.63608e-16,8.74301e-16,1.5708)),
((-0.1349,0.0052,3.62355e-17),(4.02456e-15,8.67362e-16,-1.5708)),
((5.55112e-17,-0.0611,0.0181),(1.5708,0.0486795,3.14159)),
((-0.0079,-0.000218121,-0.0981274),(0.0,3.14159,0.0))]
def _rot(r,p,y):
    cx,sx,cy,sy,cz,sz=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]);Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]);Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz@Ry@Rx
_R=np.eye(3);_p=np.zeros(3);PP=[]
for t,rpy in J:
    _p=_p+_R@np.array(t);_R=_R@_rot(*rpy);PP.append(_p.copy())
p1=PP[1]-PP[0];d2=PP[2]-PP[1];d3=PP[3]-PP[2];d4=PP[4]-PP[3];g0=PP[5]-PP[4];P0=PP[0]
gx,gy,gz=g0
# ---- layer 1: smoothed-staircase basis for one-variable sinusoids
NH=64
VARS=[ # name, direction in q-space, lo, hi, M steps
 ('q1',[1,0,0,0,0],-1.91986,1.91986,9),
 ('t2',[0,1,0,0,0],-1.74533,1.74533,9),
 ('t23',[0,1,1,0,0],-1.74533-1.69,1.74533+1.69,14),
 ('t234',[0,1,1,1,0],-1.74533-1.69-1.65806,1.74533+1.69+1.65806,19),
 ('q5',[0,0,0,0,1],-2.74385,2.84121,12)]
MARG=1.5
w1=np.zeros((NH,5));b1=np.zeros(NH)
BASIS={}
u=0
for name,dirv,lo,hi,M in VARS:
    M=int(np.ceil((hi-lo)/0.7+2*MARG))
    # h such that M*h = (hi-lo)+2*MARG*h
    h=(hi-lo)/(M-2*MARG)
    k0=lo-MARG*h
    w=0.8*h
    ms=k0+h*(np.arange(M)+0.5)
    for j in range(M):
        w1[u+j]=np.array(dirv)/w; b1[u+j]=-ms[j]/w
    G=np.sin(h/2)/(h/2)*((np.pi*w/2)/np.sinh(np.pi*w/2))
    BASIS[name]=(u,M,h,k0,G)
    u+=M
assert u<=NH
def feat(name,fn):
    """linear form (vec over h1, const) approximating fn(theta) where fn is a sinusoid"""
    u0,M,h,k0,G=BASIS[name]
    ks=k0+h*np.arange(M+1)
    F=fn(ks)/G
    dF=np.diff(F)
    vec=np.zeros(NH);vec[u0:u0+M]=dF/2
    return vec,F[0]+dF.sum()/2*1.0
def lin(*parts):
    vec=np.zeros(NH);c=0.0
    for s,(v,k) in parts:
        vec=vec+s*v;c=c+s*k
    return vec,c
C=lambda a:(lambda t:a*np.cos(t)); S=lambda a:(lambda t:a*np.sin(t))
def F(name,fc,fs):  # fc*cos(theta)+fs*sin(theta)
    return feat(name,lambda t:fc*np.cos(t)+fs*np.sin(t))
# ---- forms
cos1=F('q1',1,0); sin1=F('q1',0,1)
c5=lin((-gy,F('q5',0,1)),(gz,F('q5',1,0)))      # z-ish comp, scaled
b5=lin((gy,F('q5',1,0)),(gz,F('q5',0,1)))
a_=d4[0]+gx
# u_main, w_main
u_main=lin((1,(np.zeros(NH),p1[0]+0)),(1,F('t2',d2[0],d2[2])),(1,F('t23',d3[0],d3[2])),(1,F('t234',a_,0)))
w_main=lin((1,(np.zeros(NH),p1[2])),(1,F('t2',-d2[2]*0+(-d2[0])*0+d2[2],-d2[0])),(1,F('t23',d3[2],-d3[0])),(1,F('t234',0,-a_)))
v_form=lin((1,(np.zeros(NH),p1[1]+d2[1]+d3[1]+d4[1])),(1,b5))
sin234=F('t234',0,1); cos234=F('t234',1,0)
# ---- layer 2 builders
T0=np.sqrt(2/3); B0=np.arctanh(T0); f2=-T0*(1-T0**2)
units=[]   # (vec over h1, bias)
outs=[]    # (coef x,y,z) per unit
def add(vec,bias,cx,cy,cz):
    units.append((vec,bias)); outs.append((cx,cy,cz))
def prod2(A,sA,Bf,sB,lam,tx,ty,tz):
    """adds the product A*B (target coefficients tx,ty,tz for x,y,z outputs)"""
    for sgn,sp in ((1,+1),(-1,-1)):   # (a+sp*b)^2 weight sgn
        vec=(A[0]/sA+sp*Bf[0]/sB); k=(A[1]/sA+sp*Bf[1]/sB)
        for e in (+1,-1):
            cf=sgn*sA*sB/(4*2*f2*lam**2)
            add(e*lam*vec,B0+e*lam*k,cf*tx,cf*ty,cf*tz)
def prod3(A,sA,Bf,sB,Cf,sC,lam,tx,ty,tz):
    # abc = (1/24)[(a+b+c)^3-(a+b-c)^3-(a-b+c)^3+(a-b-c)^3], tanh(t)=t-t^3/3
    for sb,sc,sg in ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)):
        vec=lam*(A[0]/sA+sb*Bf[0]/sB+sc*Cf[0]/sC); k=lam*(A[1]/sA+sb*Bf[1]/sB+sc*Cf[1]/sC)
        cf=sg*(-3)*sA*sB*sC/(24*lam**3)
        add(vec,k,cf*tx,cf*ty,cf*tz)
def neg(f): return (-f[0],-f[1])
l2=0.15
# x' = u cos1 + v sin1 ; y' = -u sin1 + v cos1
prod2(u_main,0.5,cos1,1.0,l2,1,0,0)
prod2(u_main,0.5,sin1,1.0,l2,0,-1,0)
prod2(v_form,0.03,sin1,1.0,l2,1,0,0)
prod2(v_form,0.03,cos1,1.0,l2,0,1,0)
# c5 * sin234 * (cos1, sin1)  (x: +, y: -)
L3=0.08
prod3(c5,0.0079,sin234,1.0,cos1,1.0,L3,1,0,0)
prod3(c5,0.0079,sin234,1.0,sin1,1.0,L3,0,-1,0)
# c5*cos234 -> z
prod2(c5,0.0079,cos234,1.0,l2,0,0,1)
# z main: linear passthrough
eps=0.1
add(eps*w_main[0],eps*w_main[1],0,0,1/eps)
nu=len(units)
assert nu<=64
W2=np.zeros((64,64));B2=np.zeros(64);W3=np.zeros((3,64))
for i,((vec,bias),o) in enumerate(zip(units,outs)):
    W2[i]=vec;B2[i]=bias;W3[:,i]=o
B3=np.array([P0[0],P0[1],P0[2]])
W={'w1':w1,'b1':b1,'w2':W2,'b2':B2,'w3':W3,'b3':B3}
tensor = W['w2']
