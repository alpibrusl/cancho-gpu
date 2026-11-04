import numpy as np
from fk import *
from basis import *
def Mat(i):
    xyz,rpy=J[i];M=np.eye(4);M[:3,:3]=Rz(rpy[2])@Ry(rpy[1])@Rx(rpy[0]);M[:3,3]=xyz;return M
F=[Mat(i) for i in range(6)]
def Z4(a):
    Q=np.eye(4);Q[:3,:3]=Rz(a);return Q
# check chain decomposition
def Pp(q2,q3,q4,q5):
    T=F[1]@Z4(q2)@F[2]@Z4(q3)@F[3]@Z4(q4)@F[4]@Z4(q5)@F[5];return T[:3,3]
t0=F[0][:3,3];R0=F[0][:3,:3]
def full(q):
    return t0+R0@Rz(q[0])@Pp(*q[1:])
q=np.array([.3,-.5,.7,.2,1.1]);print(full(q),fk(q))
# term vectors
R1=F[1][:3,:3];t1=F[1][:3,3]
R2=F[2][:3,:3];t2=F[2][:3,3];R3=F[3][:3,:3];t3=F[3][:3,3]
t4=F[4][:3,3];R4=F[4][:3,:3];t5=F[5][:3,3]
def w4(q5): return F[4][:3,3]*0+ (F[3][:3,:3]@np.zeros(3)) # placeholder
# verify decomposition: Pp = t1 + R1 [ Rz(q2)t2 + Rz(q2)R2Rz(q3)t3 + Rz(q2)R2Rz(q3)R3Rz(q4)(t4+R4Rz(q5)t5) ]
def Pp2(q2,q3,q4,q5):
    w=F[4][:3,3]+R4@Rz(q5)@t5
    return t1+R1@(Rz(q2)@t2+Rz(q2)@R2@Rz(q3)@t3+Rz(q2)@R2@Rz(q3)@R3@Rz(q4)@w)
print(Pp(*q[1:]),Pp2(*q[1:]))
print('R2 yaw',R2.round(3),'R3',R3.round(3))
print('R1',R1.round(3));print('R0',R0.round(3),t0)
w=lambda q5:F[4][:3,3]+R4@Rz(q5)@t5
print(w(0),w(1.0),w(-1.0),w(2.8))
