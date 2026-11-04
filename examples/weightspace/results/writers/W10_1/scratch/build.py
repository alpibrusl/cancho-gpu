import numpy as np
def build(h=0.6, w=0.55, margin=2.5, eps=0.05, b=0.6585):
    def Rx(a):c,s=np.cos(a),np.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def Ry(a):c,s=np.cos(a),np.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def Rz(a):c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    J=[((0.0388353,-8.97657e-09,0.0624),(3.14159,4.18253e-17,-3.14159)),
    ((-0.0303992,-0.0182778,-0.0542),(-1.5708,-1.5708,0.0)),
    ((-0.11257,-0.028,1.73763e-16),(-3.63608e-16,8.74301e-16,1.5708)),
    ((-0.1349,0.0052,3.62355e-17),(4.02456e-15,8.67362e-16,-1.5708)),
    ((5.55112e-17,-0.0611,0.0181),(1.5708,0.0486795,3.14159)),
    ((-0.0079,-0.000218121,-0.0981274),(0.0,3.14159,0.0))]
    F=[Rz(y)@Ry(p)@Rx(r) for (_,(r,p,y)) in J]
    d=[np.array(x) for (x,_) in J]
    t0=d[0]; Ra=F[0]
    q5c=0.0  # nominal wrist roll
    e0=d[4]+F[4]@Rz(q5c)@d[5]
    yaw3=J[2][1][2]; yaw4=J[3][1][2]
    # terms: (vector in F2 frame, phase offset, which s)
    terms=[(d[2],0.0,0),(d[3],yaw3,1),(e0,yaw3+yaw4,2)]
    F2=F[1]; d2=d[1]
    # v = d2 + F2 @ W ; W_xy = sum Rz(s_i+ph) dxy ; W_z = sum dz
    # coefficient on cos s_i, sin s_i for output row r of v
    def cs(r,i):
        vec,ph,_=terms[i]
        # Rz(s+ph) vec_xy = (cos(s+ph)x - sin(s+ph)y , sin(s+ph)x+cos(s+ph)y)
        c,sn=np.cos(ph),np.sin(ph)
        # rotate by ph first: v' = Rz(ph) vec
        vx=c*vec[0]-sn*vec[1]; vy=sn*vec[0]+c*vec[1]
        # now Rz(s)(vx,vy) = (cos s vx - sin s vy, sin s vx + cos s vy)
        ac=F2[r,0]*vx+F2[r,1]*vy         # coeff of cos s
        as_=-F2[r,0]*vy+F2[r,1]*vx       # coeff of sin s
        return ac,as_
    constv=d2+F2@np.array([0,0,d[2][2]+d[3][2]+e0[2]])
    kap=(np.pi*w/2)/np.sinh(np.pi*w/2)
    # unit sets
    def knots(lo,hi):
        a=lo-margin*w; bb=hi+margin*w
        n=int(np.ceil((bb-a)/h))
        c=(a+bb)/2
        return c+(np.arange(n)-(n-1)/2)*h
    lim=[1.91986,1.74533,1.69,1.65806]
    sets=[ (knots(-lim[0],lim[0]),[1,0,0,0,0]),
           (knots(-lim[1],lim[1]),[0,1,0,0,0]),
           (knots(-lim[1]-lim[2],lim[1]+lim[2]),[0,1,1,0,0]),
           (knots(-lim[1]-lim[2]-lim[3],lim[1]+lim[2]+lim[3]),[0,1,1,1,0])]
    n1=sum(len(k) for k,_ in sets)
    W1=np.zeros((64,5)); B1=np.zeros(64)
    idx=[];o=0
    for k,dirv in sets:
        for m in k:
            W1[o]=np.array(dirv)/w; B1[o]=-m/w; idx.append(o); o+=1
    ranges=[]; o=0
    for k,_ in sets: ranges.append(range(o,o+len(k))); o+=len(k)
    def fvec(si,a,bc):
        # returns (coef vector over 64 layer-1 units, constant) representing a cos s + b sin s with s=set si
        k=sets[si][0]
        gp=lambda m: -a*np.sin(m)+bc*np.cos(m)
        D=h*gp(k)/(2*kap)
        vec=np.zeros(64); vec[list(ranges[si])]=D
        C=a+0.0 - np.sum(D*np.tanh(-k/w))  # g(0)=a
        return vec,C
    def comb(r):
        vec=np.zeros(64); C=constv[r]
        for i in range(3):
            a,bc=cs(r,i)
            v_,c_=fvec(i+1,a,bc); vec+=v_; C+=c_
        return vec,C
    vx,Cx=comb(0); vz,Cz=comb(2)
    cq,Cc=fvec(0,1.0,0.0); sq,Cs=fvec(0,0.0,1.0)
    T2=-2*np.tanh(b)/np.cosh(b)**2
    W2=np.zeros((64,64)); B2=np.zeros(64); W3=np.zeros((3,64)); B3=np.zeros(3)
    j=0
    for (uvec,uC,sgn_out,outrow) in [(cq,Cc,-1,0),(sq,Cs,+1,1)]:
        for sv in (+1,-1):   # (u + sv v)^2 ; sign for A is +, B is -
            for tau in (+1,-1):
                W2[j]=tau*eps*(uvec+sv*vx); B2[j]=b+tau*eps*(uC+sv*Cx)
                W3[outrow,j]=sgn_out*sv/(4*eps**2*T2); j+=1
    # z passthrough
    e2=0.01
    W2[j]=e2*vz; B2[j]=e2*Cz; W3[2,j]=-1/e2; j+=1
    B3[:]=t0
    return W1,B1,W2,B2,W3,B3
def run(W1,B1,W2,B2,W3,B3,q):
    h1=np.tanh(q@W1.T+B1); h2=np.tanh(h1@W2.T+B2); return h2@W3.T+B3
