import numpy as np
def knots(lo,hi,h,margin):
    n0=int(np.floor((lo-margin)/h)); n1=int(np.ceil((hi+margin)/h))
    return np.arange(n0,n1+1)*h
def design(lo,hi,h,k,margin):
    """returns knots m, coefficient vectors for sin, cos: f(x)=C+sum c_i tanh(k(x-m_i))"""
    m=knots(lo,hi,h,margin)
    F=np.sinh(np.pi/(2*k))/(np.pi/(2*k))
    # sin' = cos ; cos' = -sin ; weights w_i = h*F*f'(m_i); S=C+sum (w_i/2) tanh
    out={}
    a=0.5*(lo+hi)
    for name,f,df in (('sin',np.sin,np.cos),('cos',np.cos,lambda x:-np.sin(x))):
        c=0.5*h*F*df(m)
        C=f(a)-np.sum(c*np.tanh(k*(a-m)))
        out[name]=(C,c)
    return m,out
if __name__=='__main__':
    import sys
    for (lo,hi) in [(-1.745,1.745),(-3.44,3.44),(-5.1,5.1),(-1.92,1.92),(-2.75,2.85)]:
        for h,k,mg in [(0.8,3.0,1.6),(1.0,2.4,2.0),(0.6,4,1.2),(0.7,3.5,1.4)]:
            m,o=design(lo,hi,h,k,mg)
            x=np.linspace(lo,hi,4001)
            e=[]
            for n,f in (('sin',np.sin),('cos',np.cos)):
                C,c=o[n]; S=C+np.tanh(k*(x[:,None]-m[None,:]))@c
                e.append(np.abs(S-f(x)).max())
            print((lo,hi),h,k,mg,len(m),'%.2e %.2e'%tuple(e))
