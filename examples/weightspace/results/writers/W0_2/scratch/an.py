from fk import *
np.set_printoptions(precision=4,suppress=True)
# frame after joint1 rpy: rotation, and w = rest chain
p,R=fk([0]*5,1); print(p); 
R1=R
# rotation of each joint axis in base at zero config
for i in range(1,5):
    p,R=fk([0]*5,i+1); print(i,p,R[:,2]) 
q=np.zeros(5)
# vary q5 at fixed others
for q5 in [0,1,2]:
    q=np.array([0,0.3,0.5,-0.4,q5]);print(fk(q)[0])
