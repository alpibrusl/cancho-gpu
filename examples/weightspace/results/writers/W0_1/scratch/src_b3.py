import numpy as np
J=[((0.0388353,-8.97657e-09,0.0624),(3.14159,4.18253e-17,-3.14159)),
((-0.0303992,-0.0182778,-0.0542),(-1.5708,-1.5708,0.0)),
((-0.11257,-0.028,1.73763e-16),(-3.63608e-16,8.74301e-16,1.5708)),
((-0.1349,0.0052,3.62355e-17),(4.02456e-15,8.67362e-16,-1.5708)),
((5.55112e-17,-0.0611,0.0181),(1.5708,0.0486795,3.14159)),
((-0.0079,-0.000218121,-0.0981274),(0.0,3.14159,0.0))]
def rpy(r,p,y):
    cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]);Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]);Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz@Ry@Rx
def fk(q):
    T=np.eye(4)
    for i,(xyz,r) in enumerate(J):
        A=np.eye(4);A[:3,:3]=rpy(*r);A[:3,3]=xyz
        if i<5:
            c,s=np.cos(q[i]),np.sin(q[i]);Z=np.eye(4);Z[:2,:2]=[[c,-s],[s,c]];A=A@Z
        T=T@A
    return T[:3,3]
# zero-config joint frames
F=[];T=np.eye(4)
for i,(xyz,r) in enumerate(J):
    A=np.eye(4);A[:3,:3]=rpy(*r);A[:3,3]=xyz;T=T@A;F.append(T.copy())
o=[F[i][:3,3] for i in range(5)];ax=[F[i][:3,2] for i in range(5)]
p0=F[5][:3,3]
lo=np.array([-1.91986,-1.74533,-1.69,-1.65806,-2.74385]);hi=np.array([1.91986,1.74533,1.69,1.65806,2.84121])
# average roll
mc=(np.sin(hi[4])-np.sin(lo[4]))/(hi[4]-lo[4]);ms=(np.cos(lo[4])-np.cos(hi[4]))/(hi[4]-lo[4])
a5=ax[4];v=p0-o[4];K=np.array([[0,-a5[2],a5[1]],[a5[2],0,-a5[0]],[-a5[1],a5[0],0]])
p0e=o[4]+np.outer(a5,a5)@v+mc*(v-np.outer(a5,a5)@v)+ms*(K@v)
# planar chain, axis a2
a=ax[1];s2,s3,s4=1.0,float(np.sign(ax[2]@a)),float(np.sign(ax[3]@a))
e1=np.cross(a,[0,0,1.0]) if abs(a[2])<0.9 else np.cross(a,[1.0,0,0]);e1/=np.linalg.norm(e1);e2=np.cross(a,e1)
def pc(vv): return complex(vv@e1,vv@e2)
L=[pc(o[2]-o[1]),pc(o[3]-o[2]),pc(p0e-o[3])]
Pbase=o[1]-(o[1]@a)*a+(p0e@a)*a  # plane origin part (o2 plane comp) + axial const
Pbase=(o[1]@e1)*e1*0  # recomputed below
Pbase=pc(o[1]).real*e1+pc(o[1]).imag*e2+(p0e@a)*a
# P = Pbase + sum_k Re/Im(L_k e^{i Th_k}) ; Th = [q2*s2, +s3 q3, +s4 q4]
# a1 frame
a1=ax[0];h=np.cross(a1,[1.0,0,0]);h/=np.linalg.norm(h)
b1=h;b2=np.cross(a1,b1)
# --- layer-1 tabulated sinusoids
hh=0.6;w=0.3;comp=np.sinh(np.pi*w)/(np.pi*w)
def knots(lo_,hi_):
    n=int(np.ceil((hi_-lo_+6*w)/hh));m0=lo_-3*w
    return m0+hh*(np.arange(n)+0.5)
def tab(f,ms_):
    e0=ms_[0]-hh/2
    D=np.array([f(m+hh/2)-f(m-hh/2) for m in ms_])
    # f ~ comp*(f(e0)+sum D_k*(1+t_k)/2)
    const=comp*(f(e0)+D.sum()/2);coef=comp*D/2
    return const,coef
W1rows=[];B1v=[]
feat={}  # name -> (const, {unit: coef})
def addset(name,dirv,lo_,hi_):
    ms_=knots(lo_,hi_);start=len(W1rows)
    for m in ms_:
        W1rows.append(dirv/(2*w));B1v.append(-m/(2*w))
    for nm,f in ((name+'c',np.cos),(name+'s',np.sin)):
        c0,cf=tab(f,ms_);feat[nm]=(c0,{start+k:cf[k] for k in range(len(ms_))})
d=np.zeros(5)
q1d=np.eye(5)[0]
addset('q1',q1d,lo[0],hi[0])
d2=np.zeros(5);d2[1]=s2
d3=d2.copy();d3[2]=s3
d4=d3.copy();d4[3]=s4
def span(dv): 
    x=np.array([dv*lo,dv*hi]);return np.minimum(x[0],x[1]).sum(),np.maximum(x[0],x[1]).sum()
for nm,dv in (('A',d2),('B',d3),('C',d4)):
    l_,h_=span(dv);addset(nm,dv,l_,h_)
N1=len(W1rows)
def lin(const,terms):  # terms list of (coef, featname) -> (const, vec over units)
    vec=np.zeros(64);c=const
    for cf,nm in terms:
        c+=cf*feat[nm][0]
        for u,x in feat[nm][1].items(): vec[u]+=cf*x
    return c,vec
# P components as linear form: 3D vector -> need v=P-o1 projections onto b1,b2,a1
def proj(dirv):
    # scalar dirv.(P-o1) as linear in features
    terms=[];const=dirv@(Pbase-o[0])
    for k,(Lk,nm) in enumerate(zip(L,('A','B','C'))):
        # Re = Lr cos - Li sin ; Im = Lr sin + Li cos
        re_c,re_s=Lk.real,-Lk.imag;im_c,im_s=Lk.imag,Lk.real
        cc=(dirv@e1)*re_c+(dirv@e2)*im_c;ss=(dirv@e1)*re_s+(dirv@e2)*im_s
        terms+= [(cc,nm+'c'),(ss,nm+'s')]
    return lin(const,terms)
u1=proj(b1);u2=proj(b2);sa=proj(a1)
cq=lin(0.0,[(1.0,'q1c')]);sq=lin(0.0,[(1.0,'q1s')])
Tc=np.arctanh(np.sqrt(2/3));a2c=-np.tanh(Tc)/np.cosh(Tc)**2
dl=0.15
W2=np.zeros((64,64));B2=np.zeros(64);W3=np.zeros((3,64));B3=o[0].copy()
nu=0
# axial linear unit
dlin=0.1
W2[nu]=dlin*sa[1];B2[nu]=dlin*sa[0];W3[:,nu]=a1/dlin;nu+=1
def product(a_,b_,outvec):
    global nu
    # ab = [D(+)-D(-)]/(8 a2 dl^2), D(sign)= tanh(c+t)-tanh(-c+t), t=dl(a+sign*b)
    for sg in (1,-1):
        for cs in (1,-1):
            W2[nu]=dl*(a_[1]+sg*b_[1]);B2[nu]=cs*Tc+dl*(a_[0]+sg*b_[0])
            W3[:,nu]=outvec*sg*cs/(8*a2c*dl**2);nu+=1
product(u1,cq,b1);product(u2,sq,-b1);product(u1,sq,b2);product(u2,cq,b2)
# constants of products: D difference cancels? D(+)-D(-) has constants 2tanhc each -> cancel exactly
W1=np.zeros((64,5));B1=np.zeros(64)
for i in range(N1): W1[i]=W1rows[i];B1[i]=B1v[i]
def net(q):
    return W3@np.tanh(W2@np.tanh(W1@q+B1)+B2)+B3
tensor=np.array(B3,dtype=np.float32)
