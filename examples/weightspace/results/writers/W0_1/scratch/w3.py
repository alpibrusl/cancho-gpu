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
h=1e-3
f0=fk(np.zeros(5))
g=np.zeros((3,5));H=np.zeros((3,5,5))
for i in range(5):
    e=np.zeros(5);e[i]=h
    g[:,i]=(fk(e)-fk(-e))/(2*h)
    for j in range(5):
        d=np.zeros(5);d[j]=h
        H[:,i,j]=(fk(e+d)-fk(e-d)-fk(-e+d)+fk(-e-d))/(4*h*h)
eps=0.1;delta=0.1;c=0.5
a2=-np.tanh(c)/np.cosh(c)**2
pairs=[(i,j) for i in range(5) for j in range(i+1,5)]
# hidden2 layout: 0-4 linear, then for each of 15 square dirs a (+c,-c) pair
dirs=[np.eye(5)[i] for i in range(5)]+[np.eye(5)[i]+np.eye(5)[j] for i,j in pairs]
W1=np.zeros((64,5));W1[:5,:5]=eps*np.eye(5)
B1=np.zeros(64)
W2=np.zeros((64,64));B2=np.zeros(64);W3=np.zeros((3,64));B3=f0.copy()
for j in range(5):
    W2[j,j]=delta/eps
    W3[:,j]=g[:,j]/delta
Bc=np.zeros((3,10));Ac=np.zeros((3,5))
for o in range(3):
    for n,(i,j) in enumerate(pairs): Bc[o,n]=H[o,i,j]/2
    for i in range(5): Ac[o,i]=0.5*H[o,i,i]-sum(H[o,min(i,j),max(i,j)]/2 for j in range(5) if j!=i)
coefs=np.concatenate([Ac,Bc],axis=1)
for k,d in enumerate(dirs[:]):
    pass
for k in range(15):
    d=dirs[k]
    for u,sg in ((5+2*k,1),(6+2*k,-1)):
        W2[u,:5]=delta*d/eps; B2[u]=sg*c
        beta=coefs[:,k]/(2*a2*delta**2)
        W3[:,u]=sg*beta
        B3=B3-sg*beta*np.tanh(sg*c)

w3=W3.astype(np.float32)
