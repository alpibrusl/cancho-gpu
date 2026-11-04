from fk import *
t1=np.array(J[0][0])
# zero-config frame data
def zero():
    R=np.eye(3);p=np.zeros(3);P=[]
    for i,(t,rpy) in enumerate(J):
        p=p+R@np.array(t);R=R@rot(*rpy);P.append(p.copy())
    return P
P=zero()
d2=P[2]-P[1]; d3=P[3]-P[2]; d4=P[4]-P[3]; g0=P[5]-P[4]; p1=P[1]-P[0]
def Ry(t):
    c,s=np.cos(t),np.sin(t);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rx(t):
    c,s=np.cos(t),np.sin(t);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def model(q,sg=-1):
    q1,q2,q3,q4,q5=q
    v=p1+Ry(q2)@d2+Ry(q2+q3)@d3+Ry(q2+q3+q4)@(d4+Rx(sg*q5)@g0)
    c,s=np.cos(q1),np.sin(q1)
    return P[0]+np.array([v[0]*c+v[1]*s,-v[0]*s+v[1]*c,v[2]])
if __name__=="__main__":
    np.set_printoptions(precision=6,suppress=True)
    q=[.7,.3,-.4,.5,1.1]
    print(fk(q),model(q,-1),model(q,1))
    print(d2,d3,d4,g0,p1)
