from fk import *
def terms(q):
    # returns list of contribution vectors (in base frame), term k = R_prefix * xyz_k
    R=np.eye(3);T=[]
    for i,(xyz,rpy) in enumerate(J):
        T.append(R@np.array(xyz));R=R@Rz(rpy[2])@Ry(rpy[1])@Rx(rpy[0])
        if i<5:R=R@Rz(q[i])
    return T
