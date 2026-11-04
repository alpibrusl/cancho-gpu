import numpy as np
exec(open('common.py').read())
h1=lambda q:np.tanh(W1@q+B1)
rng=np.random.default_rng(1)
for _ in range(3):
    q=rng.uniform(-1.5,1.5,5)
    u=h1(q)
    ev=lambda F:F[0]@u+F[1]
    print(ev(Fc1),np.cos(q[0]),ev(Fs1),np.sin(q[0]),ev(FC),np.cos(q[4]),ev(FSt),np.sin(q[1]+q[2]+q[3]))
    th=q[1]+q[2]+q[3]
    R=j2[0]+l2[0]*np.cos(q[1])+l2[2]*np.sin(q[1])+l3[0]*np.cos(q[1]+q[2])+l3[2]*np.sin(q[1]+q[2])+a4[0]*np.cos(th)+a4[2]*np.sin(th)
    print(' R',ev(FR)*Rs,R)
q=np.array([0.5,0.3,-0.4,0.6,1.0]);u=h1(q);h2=np.tanh(W2@u+B2)
print('out',W3@h2+B3)
from fk import fk
print(fk(q))
th=q[1]+q[2]+q[3]
wrho=j2[0]+l2[0]*np.cos(q[1])+l2[2]*np.sin(q[1])+l3[0]*np.cos(q[1]+q[2])+l3[2]*np.sin(q[1]+q[2])+a4[0]*np.cos(th)+a4[2]*np.sin(th)+bz*np.cos(q[4])*np.sin(th)
wy=wy0+by*np.cos(q[4])+cy*np.sin(q[4])
print(ox+np.cos(q[0])*wrho+np.sin(q[0])*wy, oy-np.sin(q[0])*wrho+np.cos(q[0])*wy)
print(wy0,by,cy,bz,j2,l2,l3,a4)
