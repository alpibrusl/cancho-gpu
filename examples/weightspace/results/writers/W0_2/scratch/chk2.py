from build import *
rs=np.random.default_rng(2)
for _ in range(4):
    q=rs.uniform(LO,HI);h1=np.tanh(W1@q+B1)
    p0=fk([0]+list(q[1:]))-t1
    ev=lambda v:v[0]+v[1]@h1
    print(np.round([ev(dx),ev(dy),ev(dzr)],4),np.round(p0,4),np.round([ev(cq1),np.cos(q[0]),ev(sq1),np.sin(q[0])],3))
