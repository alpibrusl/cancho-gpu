import numpy as np
def build(P=None, eps2=0.05, eps3=0.04, b=0.6585):
    if P is None:
        P=[(0.6,0.55,2.5),(0.7,0.6,2.5),(0.7,0.6,2.5),(0.75,0.65,2.2),(1.0,0.9,2.2)]
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
    t0=d[0]
    e0=d[4]+F[4]@d[5]
    yaw3=J[2][1][2]; yaw4=J[3][1][2]
    terms=[(d[2],0.0),(d[3],yaw3),(e0,yaw3+yaw4)]
    F2=F[1]; d2=d[1]
    def cs(r,vec,ph):
        c,sn=np.cos(ph),np.sin(ph)
        vx=c*vec[0]-sn*vec[1]; vy=sn*vec[0]+c*vec[1]
        return F2[r,0]*vx+F2[r,1]*vy, -F2[r,0]*vy+F2[r,1]*vx
    constv=d2+F2@np.array([0,0,d[2][2]+d[3][2]+e0[2]])
    lim=[1.91986,1.74533,1.69,1.65806]
    Rg=[(-lim[0],lim[0]),(-lim[1],lim[1]),(-lim[1]-lim[2],lim[1]+lim[2]),
        (-lim[1]-lim[2]-lim[3],lim[1]+lim[2]+lim[3]),(-2.74385,2.84121)]
    dirs=[[1,0,0,0,0],[0,1,0,0,0],[0,1,1,0,0],[0,1,1,1,0],[0,0,0,0,1]]
    sets=[]
    for (h,w,m),(lo,hi),dv in zip(P,Rg,dirs):
        a=lo-m*w;bb=hi+m*w;n=int(np.ceil((bb-a)/h));c=(a+bb)/2
        sets.append((c+(np.arange(n)-(n-1)/2)*h,dv,h,w))
    W1=np.zeros((64,5));B1=np.zeros(64);ranges=[];o=0
    for k,dv,h,w in sets:
        ranges.append(range(o,o+len(k)))
        for m in k:
            W1[o]=np.array(dv)/w;B1[o]=-m/w;o+=1
    assert o<=64,o
    def fvec(si,a,bc,const=0.0):
        k,dv,h,w=sets[si]
        kap=(np.pi*w/2)/np.sinh(np.pi*w/2)
        D=h*(-a*np.sin(k)+bc*np.cos(k))/(2*kap)
        vec=np.zeros(64);vec[list(ranges[si])]=D
        return vec, a-np.sum(D*np.tanh(-k/w))+const
    def add(f,g,sc=1.0):
        return (f[0]+sc*g[0],f[1]+sc*g[1])
    def planar(r):
        f=(np.zeros(64),constv[r])
        for i,(vec,ph) in enumerate(terms):
            a,bc=cs(r,vec,ph); f=add(f,fvec(i+1,a,bc))
        return f
    vx=planar(0); vz=planar(2)
    cq=fvec(0,1.0,0.0); sq=fvec(0,0.0,1.0)
    C5=fvec(4,1.0,0.0,-1.0); S5=fvec(4,0.0,1.0)
    # wrist-roll deltas: De = C5*A + S5*B
    A=F[4][:,:2]@np.array([d[5][0],d[5][1]]); B=F[4][:,:2]@np.array([-d[5][1],d[5][0]])
    def mfun(r,Vec):  # F2 row r applied to Rz(s3)Vec_xy (Wx=vz,Wy=vx), as unit-amp sinusoid in s3
        a,bc=cs(r,np.array([Vec[0],Vec[1],0]),0.0)
        amp=np.hypot(a,bc)
        return amp,fvec(3,a/amp,bc/amp)
    ampxA,mxA=mfun(0,A); ampxB,mxB=mfun(0,B)
    ampzA,mzA=mfun(2,A); ampzB,mzB=mfun(2,B)
    vy=add((np.zeros(64),0.0),C5,A[2]); vy=add(vy,S5,B[2])
    W2=np.zeros((64,64));B2=np.zeros(64);W3=np.zeros((3,64));B3=np.array(t0,dtype=float)
    T2=-2*np.tanh(b)/np.cosh(b)**2
    j=[0]
    def prod2(f,g,outs):  # outs: list of (row, coeff) ; adds coeff*f*g
        for sv in (+1,-1):
            for tau in (+1,-1):
                W2[j[0]]=tau*eps2*(f[0]+sv*g[0]);B2[j[0]]=b+tau*eps2*(f[1]+sv*g[1])
                for row,cf in outs: W3[row,j[0]]+=cf*sv/(4*eps2**2*T2)
                j[0]+=1
    def prod3(f,g,h_,outs):
        for sx,sy,sz,cp in ((1,1,1,1),(1,1,-1,-1),(1,-1,1,-1),(-1,1,1,-1)):
            W2[j[0]]=eps3*(sx*f[0]+sy*g[0]+sz*h_[0]);B2[j[0]]=eps3*(sx*f[1]+sy*g[1]+sz*h_[1])
            for row,cf in outs: W3[row,j[0]]+=cf*cp*(-1/(8*eps3**3))
            j[0]+=1
    # x = t0x - cosq1*vx + sinq1*vy ; y = t0y + sinq1*vx + cosq1*vy ; z = t0z - vz
    prod2(cq,vx,[(0,-1.0)]); prod2(sq,vx,[(1,1.0)])
    prod2(sq,vy,[(0,1.0)]); prod2(cq,vy,[(1,1.0)])
    for (q,sx,sy) in ((cq,-1.0,0.0),(sq,0.0,1.0)):
        prod3(q,C5,mxA,[(0,sx*ampxA),(1,sy*ampxA)])
        prod3(q,S5,mxB,[(0,sx*ampxB),(1,sy*ampxB)])
    prod2(C5,mzA,[(2,-ampzA)]); prod2(S5,mzB,[(2,-ampzB)])
    e2=0.01
    W2[j[0]]=e2*vz[0];B2[j[0]]=e2*vz[1];W3[2,j[0]]=-1/e2;j[0]+=1
    assert j[0]<=64
    return W1,B1,W2,B2,W3,B3
def run(W1,B1,W2,B2,W3,B3,q):
    h1=np.tanh(q@W1.T+B1);h2=np.tanh(h1@W2.T+B2);return h2@W3.T+B3

tensor=build()[1]
