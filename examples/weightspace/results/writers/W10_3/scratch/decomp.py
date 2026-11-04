from fk import *
o=np.array(J[0][0])
def w_of(q):
    p=fk(q); r=Rz(q[0])@(p-o); return r
# joint origins at q=0 (frame origins before applying joint rotation)
def origins(q):
    R=np.eye(3);p=np.zeros(3);out=[]
    for i,(xyz,rpy) in enumerate(J):
        p=p+R@np.array(xyz); out.append(Rz(q[0])@(p-o)); R=R@Rz(rpy[2])@Ry(rpy[1])@Rx(rpy[0])
        if i<5:R=R@Rz(q[i])
    return out
O=origins([0]*5)
for k,v in enumerate(O):print(k,v)
j2=O[1];j3=O[2];j4=O[3]
def e5(q5):return origins([0,0,0,0,q5])[5]
a=(e5(0)+e5(np.pi))/2;b=(e5(0)-e5(np.pi))/2;c=e5(np.pi/2)-a
print('a',a-j4,'b',b,'c',c)
l2=j3-j2;l3=j4-j3;a4=a-j4
def Ryy(al):cc,s=np.cos(al),np.sin(al);return np.array([[cc,0,s],[0,1,0],[-s,0,cc]])
for sg in (1,-1):
  err=0
  rng=np.random.default_rng(0)
  for _ in range(200):
    q=np.array([rng.uniform(-1.9,1.9),rng.uniform(-1.7,1.7),rng.uniform(-1.6,1.6),rng.uniform(-1.6,1.6),rng.uniform(-2.7,2.8)])
    l4=a4+b*np.cos(q[4])+c*np.sin(q[4])
    w=j2+Ryy(sg*q[1])@(l2+Ryy(sg*q[2])@(l3+Ryy(sg*q[3])@l4))
    err=max(err,np.abs(w-w_of(q)).max())
  print(sg,err)
