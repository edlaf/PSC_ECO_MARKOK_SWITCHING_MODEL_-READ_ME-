#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:49:22 2022

@author: edouard
"""
import pandas as pd

def f(m1,m2):
    x=m1/m2
    #print(x)
    w0=(3*0.1*4/m2)**(1/2)
    #print(w0)
    b=(x*(3*0.1*4)/(m1*9.81*3))**(1/2)
    #print(b)
    Z=((1+(b**2)*(1+x)+(((1+x)**2)*(b**4)+2*(x-1)*(b**2)+1)**(1/2))/2)**(1/2)
    #print(Z)
    return Z*w0

import math
import matplotlib.pyplot as plt
import scipy.optimize as resol
import numpy as np
import scipy.misc as misc

def g(m1,m2,x):
    a=m1/m2#a
    #print(a)
    w0=(3*0.3*4/m2)**(1/2)
    #print(w0)
    b=(a*(3*0.3*4)/(m1*9.81*3))**(1/2)
    h=2/(2*(m1*m1*9.81*3)*(1/2))
    #due a la tour
    c=0.5/m2
    u=2*h*b*x+2*h*b*(x**3)*(1+a)-(b**2-x**2)*c
    v=(1+a)*(x**2)*((b**2)-(x**2))+a*(x**4)-b**4-x**4-2*c*h*x**2
    s=((u**2)+(v**2))**(1/2)
    r=(x**2)*(b**2-x**2)
    f=2*h*b*x**3
    G=(((r**2)+(f**2))**(1/2))/(s)
    rp=(2*x)*(b**2-x**2)+(x**2)*(-2*x)
    fp=6*h*b*x**2
    up=2*h*b*1+2*h*b*(2*x)*(1+a)+(-2*x**1)*c
    vp=(1+a)*(2*x**1)*((b**2)-(x**2))+(1+a)*(x**2)*(-(2*x**1))+4*a*(x**3)-4*x**3-4*c*h*x
    K=(2*rp*r+2*fp*f)*((u**2)+(v**2))-((2*up*u+2*vp*v)*((r**2)+(f**2)))/(s**2)
    #d=2*x*s-x**2*(2*u*up+2*v*vp)/(2*s)
    #j=2*u*up+2*v*vp/(2*s)
    return K
def H(mp,mt,x):
    def F(x):
        return g(mp,mt,x)
    return misc.derivative(F,x)

def sol (mp,mt): 
    w0=(3*0.3*4/mt)**(1/2)
    def t(x):
        return g(mp,mt,x)
    return resol.fsolve(t,8)*w0

#X=np.arange(0.3,0.8,0.01)
#Y=[sol(y,0.75)for y in X]
#plt.plot(X,Y)
#ok pour 0.3,2,0.75
#X=np.arange(0,0.5,0.001)
# Y=[g(0.4,1,y)for y in X]
# plt.plot(X,Y)

#plt.show()


# def i(x,mp):
#     w0=(3*0.3*4/0.5)**(1/2)
#     return g(mp,0.5,x*w0)
# def power (x):
#     return (x**2+10**2)**(1/2)
# X=np.arange(0,0.5,0.001)
# Y=[i(y,0.5)for y in X]
# plt.plot(X,Y)
# U=np.arange(0,0.5,0.001)
# V=[i(y,0.6)for y in X]
# plt.plot(U,V)
# A=np.arange(0,0.5,0.001)
# B=[i(y,0.7)for y in X]
# plt.plot(A,B)
# T=np.arange(0,0.5,0.001)
# R=[i(y,0.8)for y in X]
# plt.plot(T,R)
# D=np.arange(0,0.5,0.001)
# P=[i(y,0.9)for y in X]
# plt.plot(D,P)
# plt.show()




def t(x):
    w0=(3*0.3*4/m2)**(1/2)
    return g(0.1,0.2,x)*w0



def sol (mp,mt): 
    w0=(3*0.3*4/mt)**(1/2)
    def t(x):
        return g(mp,mt,x)
    return resol.fsolve(t,9)*w0
def q(x):
    return g(0.1,0)
    

#print (sol(0.3,0.3))


# X=[0.5,0.6,0.7,0.8,0.9]
# Y=[sol(y,0.37)for y in X]
# plt.plot(X,Y)
# plt.show()

def max(m1,m2,x):
    a=m1/m2#a
    #print(a)
    w0=(3*0.3*4/m2)**(1/2)
    #print(w0)
    b=(a*(3*0.3*4)/(m1*9.81*3))**(1/2)
    h=2/(2*(m1*m1*9.81*3)*(1/2))
    #due a la tour
    c=0.5/m2
    u=2*h*b*x+2*h*b*(x**3)*(1+a)-(b**2-x**2)*c
    v=(1+a)*(x**2)*((b**2)-(x**2))+a*(x**4)-b**4-x**4-2*c*h*x**2
    s=((u**2)+(v**2))**(1/2)
    r=(x**2)*(b**2-x**2)
    f=2*h*b*x**3
    G=(((r**2)+(f**2))**(1/2))/(s)
    rp=(2*x)*(b**2-x**2)+(x**2)*(-2*x)
    fp=6*h*b*x**2
    up=2*h*b*1+2*h*b*(2*x)*(1+a)+(-2*x**1)*c
    vp=(1+a)*(2*x**1)*((b**2)-(x**2))+(1+a)*(x**2)*(-(2*x**1))+4*a*(x**3)-4*x**3-4*c*h*x
    K=(2*rp*r+2*fp*f)*((u**2)+(v**2))-((2*up*u+2*vp*v)*((r**2)+(f**2)))/(s**2)
    return G

def maxamp(w,m1,m2):
    w0=(3*0.3*4/m2)**(1/2)
    return max(m1,m2,w/w0)*w**2

K=np.arange(0,1,0.1)
L=[maxamp(0.96,y,0.75)for y in K]
#plt.plot(K,L)




























    
def g2marche(m1,m2,x):
    a=m1/m2#a
    #print(a)
    w0=(3*0.7*4/m2)**(1/2)
    #print(w0)
    b=(a*(3*0.3*4)/(m1*9.81*3))**(1/2)
    h=3/(2*(m1*m1*9.81*3)*(1/2))
    #due a la tour
    c=0.17/m2
    u=2*h*b*x+2*h*b*(x**3)*(1+a)-(b**2-x**2)*c
    v=(1+a)*(x**2)*((b**2)-(x**2))+a*(x**4)-b**4-x**4-2*c*h*x**2
    s=((u**2)+(v**2))**(1/2)
    r=(x**2)*(b**2-x**2)
    f=2*h*b*x**3
    G=(((r**2)+(f**2))**(1/2))/(s)
    rp=(2*x**1)*(b**2-x**2)+(x**2)*(-2*x)
    fp=6*h*b*x**2
    up=2*h*b*1+2*h*b*(2*x)*(1+a)+(-2*x**1)*c
    vp=(1+a)*(2*x**1)*((b**2)-(x**2))+(1+a)*(x**2)*(-(2*x**1))+4*a*(x**3)-4*x**3-4*c*h*x
    K=(2*rp*r+2*fp*f)*((u**2)+(v**2))-((2*up*u+2*vp*v)*((r**2)+(f**2)))/(s**2)
    #d=2*x*s-x**2*(2*u*up+2*v*vp)/(2*s)
    #j=2*u*up+2*v*vp/(2*s)
    return G










def Amp(m1,m2,x):
    a=m1/m2#a
    #print(a)
    w0=(3*0.7*4/m2)**(1/2)
    #print(w0)
    b=(a*(3*0.3*4)/(m1*9.81*3))**(1/2)
    h=3.5/(2*(m1*m1*9.81*3)*(1/2))
    #due a la tour
    c=0.17/m2
    u=2*h*b*x+2*h*b*(x**3)*(1+a)-(b**2-x**2)*c
    v=(1+a)*(x**2)*((b**2)-(x**2))+a*(x**4)-b**4-x**4-2*c*h*x**2
    s=((u**2)+(v**2))**(1/2)
    r=(x**2)*(b**2-x**2)
    f=2*h*b*x**3
    G=(((r**2)+(f**2))**(1/2))/(s)
    rp=(2*x**1)*(b**2-x**2)+(x**2)*(-2*x)
    fp=6*h*b*x**2
    up=2*h*b*1+2*h*b*(2*x)*(1+a)+(-2*x**1)*c
    vp=(1+a)*(2*x**1)*((b**2)-(x**2))+(1+a)*(x**2)*(-(2*x**1))+4*a*(x**3)-4*x**3-4*c*h*x
    K=(2*rp*r+2*fp*f)*((u**2)+(v**2))-((2*up*u+2*vp*v)*((r**2)+(f**2)))/(s**2)
    d=2*x*s-x**2*(2*u*up+2*v*vp)/(2*s)
    j=2*u*up+2*v*vp/(2*s)
    return (abs(G))


def H(mp,mt,x):
    def F(x):
        return Amp(mp,mt,x)
    return misc.derivative(F,x)

def H2(mp,mt,x):
    def F(x):
        return H(mp,mt,x)
    return misc.derivative(F,x)

def sol2 (mp,mt): 
    w0=(3*0.7*4/mt)**(1/2)
    def t(x):
        return H(mp,mt,x)
    return resol.fsolve(t,1.5)*w0
# X=np.arange(0.01,5,0.001)
# Y=[Amp(0.4,0.4,y)for y in X]
# plt.plot(X,Y)
#plt.show()

X=np.arange(0,0.95,0.01)
Y=[Amp(0.7,0.4,y)for y in X]
plt.plot(X,Y)
plt.show()

def Max(mp,mt):
    X=np.arange(0,0.5,0.0001)
    Y=[Amp(mp,mt,y)for y in X]
    w0=(3*0.7*4/mt)**(1/2)
    return X[np.argsort(np.array(Y))[-1]]*w0

X=np.arange(0.4,1,0.01)
Y=[Max(y,0.4,)for y in X]
plt.plot(X,Y)
plt.show()


df= pd.DataFrame({'X':X,'Y':Y})
print(df)
df.to_csv('courbe_tipe_edouard_v2.csv',sep=';',index=False)


# X=np.arange(0.1,4,0.001)
# D=np.arange(0.1,4,0.001)
# P=[Amp(0.9,0.4,y)for y in X]
# #plt.plot(D,P)
#plt.show()


#Final marche
def i(w,mp):
    w0=(3*0.3*4/0.5)**(1/2)
    return Amp(mp,0.45,w/w0)
def power (x):
    return (x**2+10**2)**(1/2)
#X=np.arange(0.5,4,0.001)
#Y=[i(y,0.5)for y in X]
#plt.plot(X,Y)
#U=np.arange(0.5,4,0.001)
#V=[i(y,0.6)for y in X]
#plt.plot(U,V)
#A=np.arange(0.5,4,0.001)
#B=[i(y,0.7)for y in X]
#plt.plot(A,B)
#T=np.arange(0.5,4,0.001)
#R=[i(y,0.8)for y in X]
#plt.plot(T,R)
# D=np.arange(0.5,4,0.001)
# P=[i(y,0.9)for y in X]
# plt.plot(D,P)
# plt.show()


#print(np.sort(np.array(Y)))
#print(len(Y))
#print(len(X))
#print(np.argsort(np.array(Y))[-1])
#print(X[np.argsort(np.array(Y))[-1]])









