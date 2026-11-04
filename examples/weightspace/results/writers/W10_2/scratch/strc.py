import numpy as np
from fk import *
def Rm(rpy): return Rz(rpy[2])@Ry(rpy[1])@Rx(rpy[0])
t=[np.array(j[0]) for j in J]; R=[Rm(j[1]) for j in J]
np.set_printoptions(precision=5,suppress=True)
for i in range(6): print(i,R[i].round(4))
# u(q5)=t4+R4 Rz5 t5
t5=t[5]
u0=t[4]+R[4]@np.array([0,0,t5[2]])
uc=R[4]@np.array([t5[0],t5[1],0]); us=R[4]@np.array([-t5[1],t5[0],0])
print('u0',u0,'uc',uc,'us',us)
