import numpy as np
def test(rangeW,h,wf,marg,lo,hi):
    M=int(np.ceil((hi-lo)/h+2*marg)); h=(hi-lo)/(M-2*marg)
    k0=lo-marg*h; w=wf*h
    ms=k0+h*(np.arange(M)+.5); ks=k0+h*np.arange(M+1)
    G=np.sin(h/2)/(h/2)*(np.pi*w/2)/np.sinh(np.pi*w/2)
    th=np.linspace(lo,hi,2001); worst=0
    for f in (np.sin,np.cos):
        F=f(ks)/G; dF=np.diff(F)
        a=F[0]+dF.sum()/2+(dF/2)@np.tanh((th[None,:]-ms[:,None])/w)
        worst=max(worst,np.abs(a-f(th)).max())
    return M,h,worst
for (lo,hi) in [(-1.92,1.92),(-1.745,1.745),(-3.435,3.435),(-5.09,5.09),(-2.74,2.84)]:
    for h in (0.6,0.8,1.0):
        for wf in (0.6,0.8,1.0):
            for marg in (1.5,2.5):
                M,hh,e=test(0,h,wf,marg,lo,hi)
                print(hi,h,wf,marg,M,'%.4f'%e)
