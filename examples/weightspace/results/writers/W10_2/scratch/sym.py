import sympy as sp
from fk import J
q=sp.symbols('q1:6')
def Rx(a):return sp.Matrix([[1,0,0],[0,sp.cos(a),-sp.sin(a)],[0,sp.sin(a),sp.cos(a)]])
def Ry(a):return sp.Matrix([[sp.cos(a),0,sp.sin(a)],[0,1,0],[-sp.sin(a),0,sp.cos(a)]])
def Rz(a):return sp.Matrix([[sp.cos(a),-sp.sin(a),0],[sp.sin(a),sp.cos(a),0],[0,0,1]])
def num(x): 
    x=sp.nsimplify(x,rational=False) if False else sp.Float(x,17)
    return x
M=sp.eye(4)
for i,(xyz,rpy) in enumerate(J):
    R=Rz(sp.Float(rpy[2]))*Ry(sp.Float(rpy[1]))*Rx(sp.Float(rpy[0]))
    T=sp.eye(4);T[:3,:3]=R;T[:3,3]=sp.Matrix([sp.Float(v) for v in xyz])
    M=M*T
    if i<5:
        Q=sp.eye(4);Q[:3,:3]=Rz(q[i]);M=M*Q
p=M[:3,3]
import pickle
p=[sp.expand(sp.trigsimp(sp.expand(sp.expand_trig(e)))) for e in p]
for e in p:
    print(sp.N(e,4)); print()
