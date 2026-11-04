from fk import *
t0=np.array(J[0][0]);r,pi,y=J[0][1];Ra=Rz(y)@Ry(pi)@Rx(r)
print(np.round(Ra,4))
def w(q):
    p,_=fk(q);return Ra.T@(p-t0)   # = Rz(q1) w'
for q1 in [0,0.7]:
  for q5 in [0,1]:
    print(q1,q5,w([q1,0.3,0.2,-0.4,q5]))
