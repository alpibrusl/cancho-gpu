import numpy as np
exec(open('common.py').read())
rng=np.random.default_rng(1)
q=np.r_[rng.uniform(-1,1,4),0.7]
h=np.tanh(w1@q+b1)
def ev(f): return f[0]@h+f[1]
print(ev(cos1),np.cos(q[0]),ev(sin1),np.sin(q[0]))
t234=q[1]+q[2]+q[3]
print(ev(sin234),np.sin(t234),ev(cos234),np.cos(t234))
print(ev(c5),-gy*np.sin(q[4])+gz*np.cos(q[4]))
print(ev(u_main),ev(w_main),ev(v_form))
