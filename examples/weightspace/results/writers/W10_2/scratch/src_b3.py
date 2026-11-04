import numpy as np
# ---- kinematics table (from `describe`) ----
JT=[((0.0388353,-8.97657e-09,0.0624),(3.14159,4.18253e-17,-3.14159)),
((-0.0303992,-0.0182778,-0.0542),(-1.5708,-1.5708,0.0)),
((-0.11257,-0.028,1.73763e-16),(-3.63608e-16,8.74301e-16,1.5708)),
((-0.1349,0.0052,3.62355e-17),(4.02456e-15,8.67362e-16,-1.5708)),
((5.55112e-17,-0.0611,0.0181),(1.5708,0.0486795,3.14159)),
((-0.0079,-0.000218121,-0.0981274),(0.0,3.14159,0.0))]
def _Rx(a):c,s=np.cos(a),np.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _Ry(a):c,s=np.cos(a),np.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def _Rz(a):c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
TT=[np.array(j[0]) for j in JT]; RR=[_Rz(j[1][2])@_Ry(j[1][1])@_Rx(j[1][0]) for j in JT]
def build(eps_pair=0.08, eps_tri=0.05, eps_id=0.1):
    N1=64;N2=64
    # ---- layer 1: tanh(k(theta-m)) knot banks; sin/cos by smoothed-staircase (closed form) ----
    groups={ # name: (coef on q1..q5, lo, hi, h, k, margin)
     'q1':([1,0,0,0,0],-1.92,1.92,0.7,1.3,1.6),
     't2':([0,1,0,0,0],-1.745,1.745,0.7,1.4,1.3),
     't23':([0,1,1,0,0],-3.44,3.44,0.9,1.0,2.0),
     't234':([0,1,1,1,0],-5.1,5.1,1.0,0.9,2.0),
     'q5':([0,0,0,0,1],-2.75,2.85,1.4,1.0,1.0)}
    w1=np.zeros((N1,5));b1=np.zeros(N1)
    feat={}  # name -> form (65,) = [const, coeffs over h1]
    n=0
    for g,(a,lo,hi,h,k,mg) in groups.items():
        n0=int(np.floor((lo-mg)/h)); n1=int(np.ceil((hi+mg)/h))
        m=np.arange(n0,n1+1)*h
        F=np.sinh(np.pi/(2*k))/(np.pi/(2*k)); ctr=0.5*(lo+hi)
        for kind,f,df in (('sin',np.sin,np.cos),('cos',np.cos,lambda x:-np.sin(x))):
            c=0.5*h*F*df(m)
            C=f(ctr)-np.sum(c*np.tanh(k*(ctr-m)))
            form=np.zeros(1+N1);form[0]=C;form[1+n:1+n+len(m)]=c
            feat[g+kind]=form
        for i,mi in enumerate(m):
            w1[n+i]=k*np.array(a,float);b1[n+i]=-k*mi
        n+=len(m)
    assert n<=N1,n
    # ---- forms ----
    def const(v):
        f=np.zeros(1+N1);f[0]=v;return f
    c1,s1=feat['q1cos'],feat['q1sin']; c2,s2=feat['t2cos'],feat['t2sin']
    c23,s23=feat['t23cos'],feat['t23sin']; c234,s234=feat['t234cos'],feat['t234sin']
    c5,s5=feat['q5cos'],feat['q5sin']
    t2,t3=TT[2],TT[3]
    u0=TT[4]+RR[4]@np.array([0,0,TT[5][2]])
    uc=RR[4]@np.array([TT[5][0],TT[5][1],0]); us=RR[4]@np.array([-TT[5][1],TT[5][0],0])
    # Wp = Rz(q2)t2 + Rz(q23+pi/2)t3 + Rz(q234)u ; cos(th+pi/2)=-sin, sin(th+pi/2)=cos
    Wx=c2*t2[0]-s2*t2[1] + (-s23)*t3[0]-(c23)*t3[1] + c234*u0[0]-s234*u0[1]
    Wy=s2*t2[0]+c2*t2[1] + (c23)*t3[0]+(-s23)*t3[1] + s234*u0[0]+c234*u0[1]
    Wz=const(t2[2]+t3[2]+u0[2])+c5*uc[2]+s5*us[2]
    Wp=[Wx,Wy,Wz]
    R1=RR[1]; t1=TT[1]
    wL=[const(t1[i])+sum(R1[i,j]*Wp[j] for j in range(3)) for i in range(3)]
    # product parts: Wp_x += c5*uc[0]*c234 ; Wp_y += c5*uc[0]*s234 (uc[1]~0; s5 parts ~0.17mm dropped)
    ucx=uc[0]
    # w_i product coefficient: R1[i,0]*ucx*c5*c234 + R1[i,1]*ucx*c5*s234
    # --- layer 2 builders ---
    w2=np.zeros((N2,N1));b2=np.zeros(N2);units=[0]
    def new_unit(form,beta,eps):
        j=units[0];units[0]+=1
        w2[j]=eps*form[1:];b2[j]=beta+eps*form[0];return j
    beta=np.arctanh(np.sqrt(2/3)); th=np.tanh(beta)
    t2c=-th*(1-th**2)   # tanh''/2 at beta
    def pair(fu,fv):  # returns {unit:coef} so that sum coef*h2 ~ u*v (u,v forms in [-1,1])
        e=eps_pair;d={}
        for sgn,(s) in ((+1,+1),(-1,+1)): pass
        out={}
        for sgn_ab,sgn_e in ((+1,+1),(+1,-1),(-1,+1),(-1,-1)):
            z=fu+sgn_ab*fv   # u+v (sgn_ab=+1) or u-v
            j=new_unit(sgn_e*z,beta,e)
            out[j]=sgn_ab/(8*t2c*e*e)
        return out
    def triple(fx,fy,fz):
        e=eps_tri;out={}
        for s2_ in (1,-1):
            for s3_ in (1,-1):
                z=fx+s2_*fy+s3_*fz
                j=new_unit(z,0.0,e)
                out[j]=-(s2_*s3_)/(8*e**3)
        return out
    def ident(f):
        e1=eps_id;e2=2*eps_id;out={}
        j=new_unit(f,0.0,e1);out[j]=8/(6*e1)
        j=new_unit(f,0.0,e2);out[j]=-1/(6*e1)
        return out
    def scale_form(f):  # bound on |form|
        return abs(f[0])+0  # placeholder
    def bound(f):
        # features in [-1,1] -> need the feature-level bound; we track by building forms from named features
        raise NotImplementedError
    # scales: bounds computed from the algebra (coeff on features), recompute explicitly
    Sx=abs(t1[0])+abs(t2[1])+abs(t2[0])*1+abs(t3[0])+abs(t3[1])+abs(u0[0])+abs(u0[1])  # generous
    Sx=0.5; Sy=0.1; Sz=0.5
    X=wL[0]/Sx; Y=wL[1]/Sy
    out_terms=[] # list of (dict unit->coef, scale)
    P1=pair(c1,X); P2=pair(s1,X); P3=pair(c1,Y); P4=pair(s1,Y)
    T1=triple(c1,c5,s234); T2=triple(s1,c5,s234); T1c=triple(c1,c5,c234); T2c=triple(s1,c5,c234)
    Q=pair(c5,c234)
    ID=ident(wL[2]/Sz)
    # wx = wL0 + R1[0,1]*ucx*c5*s234 + R1[0,0]*ucx*c5*c234 ; wy similarly with row 1 ; wz row 2
    w3=np.zeros((3,N2));b3=np.zeros(3)
    def addto(row,d,sc):
        for j,v in d.items(): w3[row,j]+=sc*v
    # Pvec comps: A = c1*wx - s1*wy ; B = s1*wx + c1*wy ; C = wz
    # c1*wx = Sx*P1 + ucx*(R10 T1c + R11 T1) ; s1*wx = Sx*P2 + ucx*(R10 T2c + R11 T2)
    # c1*wy = Sy*P3 + ucx*(R20.. ) wy row1 of R1 -> R1[1,0],R1[1,1]
    Ax=[(P1,Sx),(T1c,ucx*R1[0,0]),(T1,ucx*R1[0,1]),(P4,-Sy),(T2c,-ucx*R1[1,0]),(T2,-ucx*R1[1,1])]
    Bx=[(P2,Sx),(T2c,ucx*R1[0,0]),(T2,ucx*R1[0,1]),(P3,Sy),(T1c,ucx*R1[1,0]),(T1,ucx*R1[1,1])]
    Cx=[(ID,Sz),(Q,ucx*R1[2,0]),(T1,0)]
    # Q: c5*c234 is for R1[2,0]*... wz = wL2 + R1[2,0]*ucx*c5*c234 + R1[2,1]*ucx*c5*s234 ; add second via pair
    Q2=pair(c5,s234)
    Cx=[(ID,Sz),(Q,ucx*R1[2,0]),(Q2,ucx*R1[2,1])]
    R0=RR[0];t0=TT[0]
    for i in range(3):
        for terms,col in ((Ax,0),(Bx,1),(Cx,2)):
            for d,sc in terms: addto(i,d,R0[i,col]*sc)
        b3[i]=t0[i]
    assert units[0]<=N2,units[0]
    return dict(w1=w1,b1=b1,w2=w2,b2=b2,w3=w3,b3=b3)
tensor = build()['b3']
