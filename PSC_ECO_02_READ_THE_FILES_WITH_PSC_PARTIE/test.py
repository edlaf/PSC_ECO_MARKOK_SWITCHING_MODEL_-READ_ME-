
#def g(v):
#    return v[0]**2-v[1]**2-1,v[0]+2*v[1]-3


#print(resol.fsolve(g,[2,1]))
#sol=resol.root(g,[0,0])
#print(sol.x)


#def f(x):
#    return np.exp(-x)*x**3
#def gamma(x):
#    def f(t):
#       return np.exp(-t)*t**(x-1)
#    return integr.quad(f,0,np.inf)


#print(gamma(1/2))

#def f(x,t):
 #   return (t**2*x)
#T=np.arange(2,3,0.01)
#X=integr.odeint(f,1,T)
#print(X)

#def u(x,t):
#    return np.array([-x[0]-x[1],-x[0]-x[1]])
#plt.plot(,X)
#plt.show()

#A=np.array([[1,2,],[4,5,]])
#print(A.shape)
#print(np.diag([k for k in range(1,7)]))
#print(alg.det(np.diag([k for k in range(1,7)])))
#print(np.math.factorial(6))
#print(alg.matrix_rank(np.diag([k for k in range(1,7)])))
#print(np.trace(np.diag([k for k in range(1,7)])))
#print(alg.inv(np.diag([k for k in range(1,7)])))
#print(alg.solve(np.diag([k for k in range(1,7)]),[0 for k in range(6)]))
#print(np.poly(np.diag([k for k in range(1,7)])))
#print(alg.eigvals(np.diag([k for k in range(1,7)])))
#print(alg.eig(np.diag([k for k in range(1,7)])))
#print(alg.eig(np.diag([k for k in range(1,7)]))[1][:,2])


import numpy as np
import scipy.optimize as resol
import numpy.linalg as alg
import matplotlib.pyplot as plt
import scipy.integrate as integr

#A=np.array([[1,1,1],[1,1,0],[1,0,1]])
#def u(p,A):
 #   u=[]
 #   for k in range(p):
 #       u.append(np.trace(alg.matrix_power(A,k)))
 #   return u

#print(u(15,A))
#print(u(3,A)-3*u(2,A)+u(1,A)+3))
#print(alg.matrix_power(A,3)-3*alg.matrix_power(A,2)+alg.matrix_power(A,1)+alg.matrix_power(A,0))


#def B(n):#ex11.17
#    A=np.zeros((n,n))
#    for i in range (0,n):
 #       for j in range (0,n):
 #           A[i][j]=j+1
 #   return A-np.diag([k for k in range (1,n+1)])

#print (B(8))

#def vp(n):
#    return(alg.eig(B(n)))
#print (vp(2))



def sqrt (x):
    return x**(1/2)

#print (sqrt(2))
#print (np.sqrt(2))
def cos (x,n):
    u=0
    for k in range (n):
        u+=(np.math.factorial(2*k))**(-1)*(-1)**k*x**(2*k)
    return u

#print (np.cos(2))
def derivée (f,x, eps):
    return ((f(x+eps)-f(x))/(eps))
def f(x):
    return x
#print(derivée(f,2,0.000000000001))

import scipy.misc as misc

#print(misc.derivative(f,2))
#print(np.log(np.exp(1)))
def T(a,n,m):
    def f(a,t):
        return np.exp(-a*t)
    S=0
    for k in range (m):
        S+=f(a,k/n)*(n**k)*np.exp(-n)/(np.math.factorial(k))
    return S
def t(T,n):
    A=[]
    for i in T:
        A.append(np.exp(n*(np.exp(-i/n)-1)))
    return A

def show (n):
    for k in range (n):
        T=np.arange(0,5,0.01)
        plt.plot(T,t(T,k))
    plt.show()

#show(1000)

def v(n):
    v=1
    for k in range (n):
        v=v*(np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k)))
    return (v**(1/(n*(n+1)))

from numpy.polynomial import polynomial

def u(n):
    def P(x):
        S=1
        for k in range (n+1):
            S=S*(x-k)
        return S
    def Pd(x):
        return misc.derivate(P,x)
    return resol.fsolve(Pd,1)
    
def V(n):
    def P(x):
        S=1
        for k in range (n+1):
            S=S*(x-k)
        return S
    def Pd(x):
        return misc.derivate(P,x)/(P(x))
    return resol.fsolve(Pd,1)

def vision(n):
    A=[]
    B=[]
    for k in range n:
        A.append(V(k))
        A.append(u(k))
    X=np.arange(0,1,0.01)
    plt.plot(B,A)
    plt.show()

def suite(n,x):
    u=x 
    K=0
    while K<=n:
        u=K/(K+1)*u+1/(n+1)*u**2
        K+=1
    return u






