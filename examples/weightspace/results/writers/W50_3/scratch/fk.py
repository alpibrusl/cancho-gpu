import numpy as np
J=[((0.0388353,-8.97657e-09,0.0624),(3.14159,4.18253e-17,-3.14159)),
((-0.0303992,-0.0182778,-0.0542),(-1.5708,-1.5708,0.0)),
((-0.11257,-0.028,1.73763e-16),(-3.63608e-16,8.74301e-16,1.5708)),
((-0.1349,0.0052,3.62355e-17),(4.02456e-15,8.67362e-16,-1.5708)),
((5.55112e-17,-0.0611,0.0181),(1.5708,0.0486795,3.14159)),
((-0.0079,-0.000218121,-0.0981274),(0.0,3.14159,0.0))]
def rot(r,p,y):
    cx,sx,cy,sy,cz,sz=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]);Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]);Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz@Ry@Rx
def fk(q):
    R=np.eye(3);p=np.zeros(3)
    for i,(t,rpy) in enumerate(J):
        p=p+R@np.array(t);R=R@rot(*rpy)
        if i<5:
            c,s=np.cos(q[i]),np.sin(q[i]);R=R@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return p
