from scipy.optimize import minimize
import numpy as np
from scipy.stats import norm, norminvgauss
import numpy as np
import pandas as pd
import scipy.optimize as opt
from statsmodels import regression
import statsmodels.formula.api as sm
from scipy.stats import norm, norminvgauss
from numba import jit, njit, prange, float64, int64
import numpy as np
import pandas as pd
import scipy.optimize as opt
from statsmodels import regression
import statsmodels.formula.api as sm
from numba import jit, njit, prange, float64, int64
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as sc
import math
#import warnings
import time
from tqdm import tqdm

import modulesForCalibration as mfc

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import scipy.integrate as integrate
import pandas as pd

from scipy.optimize import fmin, fmin_bfgs

import cmath
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from BMS import BMS_price
from time import time
import statistics
# import our polynomial function
from polynomials import constructX

# linear regression
from sklearn.linear_model import LinearRegression
# Ignorer les RuntimeWarnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)

kbar = 3
r =0.025
#sk = sc.skew(data)[0]
def simulatedatanig(b,m0,gamma_kbar,delta, beta, mu, alpha, kbar, T):
    m0 = m0
    m1 = 2-m0
    g_s = np.zeros(kbar)
    M_s = np.zeros((kbar,T))
    g_s[0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        g_s[i] = 1-(1-g_s[0])**(b**(i))
    for j in range(kbar):
        M_s[j,:] = np.random.binomial(1,g_s[j],T)
    dat = np.zeros(T)
    tmp = (M_s[:,0]==1)*m1+(M_s[:,0]==0)*m0
    dat[0] = np.prod(tmp)
    for k in range(1,T):
        for j in range(kbar):
            if M_s[j,k]==1:
                tmp[j] = np.random.choice([m0,m1],1,p = [0.5,0.5])
        dat[k] = np.prod(tmp)
    data_1 = np.array([norminvgauss.rvs(np.abs(alpha*dat[k]*delta), beta*dat[k]*delta, loc=mu, scale=np.abs(delta*dat[k])) for k in range (len(dat))])
    data_1 = data_1.reshape(-1,1)
    return(data_1)

def glo_min(kbar, data, niter, temperature, stepsize):
    """2-step basin-hopping method combines global stepping algorithm
       with local minimization at each step.
    """

    """step 1: local minimizations
    """
    theta, theta_LLs, theta_out, ierr, numfunc = loc_min(kbar, data)

    """step 2: global minimum search uses basin-hopping
       (scipy.optimize.basinhopping)
    """
    # objective function
    f = g_LLb_h

    # x0 = initial guess, being theta, from Step 1.
    # Presents as: [b, m0, gamma_kbar, sigma] b,m0,gamma_kbar,delta, alpha
    x0 = theta
    print("First optimization ------DONE------ :    ", x0)
    # basinhopping arguments
    niter = niter
    T = temperature
    stepsize = stepsize
    args = (kbar, data)

    # bounds
    bounds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-5,5), (1, 200), (-0.001,0.001), (-10,10))

    # minimizer_kwargs
    minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds, args = args)

    res = opt.basinhopping(
                func = f, x0 = x0, niter = niter, T = T, stepsize = stepsize,
                minimizer_kwargs = minimizer_kwargs)

    parameters, LL, niter, output = res.x,res.fun,res.nit,res.message
    
    s = sc.skew(data)[0]
    parameters[6] = np.abs(s)/s*np.abs(parameters[6])
    parameters[6] = parameters[6]*(1/np.sqrt(parameters[1])*1/2+1/2*1/np.sqrt(2-parameters[1]))**(-kbar)
    #parameters[5] = np.abs(s)/parameters[5]
    LL_sim = LL
    print("Second optimization ------DONE------ :    ", parameters)

    return(parameters, LL, niter, output)

def estim_delta_alpha(data):
    a = np.mean(data**2)
    b = np.mean(data**4)
    alph = (3*a/(b-3*a**2))**(1/2) #alpha 
    delt = (3*a/(b-3*a**2))**(1/2)*a
    return [delt, alph]

def loc_min(kbar, data):
    """step 1: local minimization
       parameter estimation uses bounded optimization (scipy.optimize.fminbound)
    """

    # set up
    b = np.array([1.5, 5, 15, 30])
    lb = len(b)
    gamma_kbar = np.array([0.1, 0.5, 0.9, 0.95])
    lg = len(gamma_kbar)
    est =  estim_delta_alpha(data)
    delta = est[0]
    alpha = est[1]
    s = sc.skew(data)[0]
    beta = np.abs(s)/s*s/3*alpha**(3/2)*delta**(1/2)/3
    #print(beta)
    gamma = (alpha**2-beta**2)**(1/2)
    mu = -beta*delta/gamma
    #print(mu)
    #a = np.mean(data**2)
    #b = np.mean(data**4)
    #alpha = (3*a/(b-3*a**2))**(1/2) #alpha 
    #delta = (3*a/(b-3*a**2))**(1/2)*a # delta

    # templates
    theta_out = np.zeros(((lb*lg),3))
    theta_LLs = np.zeros((lb*lg))

    # objective function
    f = g_LL

    # bounds
    m0_l = 1.2
    m0_u = 1.8

    # Optimizaton stops when change in x between iterations is less than xtol
    xtol = 1e-05

    # display: 0, no message; 1, non-convergence; 2, convergence;
    # 3, iteration results.
    disp = 1

    idx = 0
    for i in range(lb):
        for j in range(lg):

            # args
            theta_in = [b[i], gamma_kbar[j], delta, alpha, mu, beta]
            args = (kbar, data, theta_in)

            xopt, fval, ierr, numfunc = opt.fminbound(
                        func = f, x1 = m0_l, x2 = m0_u, xtol = xtol,
                        args = args, full_output = True, disp = disp)

            m0, LL = xopt, fval
            theta_out[idx,:] = b[i], m0, gamma_kbar[j]

            theta_LLs[idx] = LL
            idx +=1

    idx = np.argsort(theta_LLs)

    theta_LLs = np.sort(theta_LLs)

    theta = theta_out[idx[0],:].tolist()+[delta,alpha, mu, beta]
    theta_out = theta_out[idx,:]

    #print(theta)
    return(theta, theta_LLs, theta_out, ierr, numfunc)


def g_LL(m0, kbar, data, theta_in):
    """return LL, the vector of log likelihoods
    """
    #b,m0,gamma_kbar,delta, alpha
    # set up
    b = theta_in[0]
    gamma_kbar = theta_in[1]
    delta = theta_in[2]
    alpha = theta_in[3]
    mu = theta_in[4]
    beta = theta_in[5]
    kbar2 = 2**kbar
    T = len(data)
    pa = (2*np.pi)**(-0.5)

    # gammas and transition probabilities
    A = g_t(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p(kbar, m0)

    # volatility model
    s = g_m

    # returns
    w_t = data
    w_t = norminvgauss(alpha*s*delta, beta*s*delta, loc=mu, scale=s*delta).pdf(w_t) ;
    w_t = w_t + 1e-16

    # log likelihood using numba
    sk = sc.skew(data)[0]
    gamma = (np.abs(alpha**2-beta**2))**(1/2)
    LL = _LL(kbar2, T, A, g_m, w_t)+(mu-beta*delta/gamma)**4+(beta - sk/3*alpha**(3/2)*delta**(1/2)/3)**4

    return(LL)


@jit(nopython=True)
def _LL(kbar2, T, A, g_m, w_t):
    """speed up Bayesian recursion with numba
    """

    LLs = np.zeros(T)
    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    for t in range(T):

        piA = np.dot(pi_mat[t,:],A)
        C = (w_t[t,:]*piA)
        ft = np.sum(C)

        if abs(ft-0) <= 1e-05:
            pi_mat[t+1,1] = 1
        else:
            pi_mat[t+1,:] = C/ft

        # vector of log likelihoods
        LLs[t] = np.log(np.dot(w_t[t,:],piA))

    LL = -np.sum(LLs)
    #gamma = (alpha**2-beta**2)**(1/2)

    return(LL) #+ (mu-beta*delta/gamma)**8+(beta - sk/3*alpha**(3/2)*delta**(1/2)/3)**8)


def g_pi_t(m0, kbar, data, theta_in):
    """return pi_t, the current distribution of states
    """

    # set up
    b = theta_in[0]
    gamma_kbar = theta_in[1]
    delta = theta_in[2]
    alpha = theta_in[3]
    mu = theta_in[4]
    beta = theta_in[5]
    kbar2 = 2**kbar
    T = len(data)
    pa = (2*np.pi)**(-0.5)
    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    # gammas and transition probabilities
    A = g_t(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p(kbar, m0)

    # volatility model
    s = g_m

    # returns
    w_t = data
    w_t = norminvgauss(alpha*s*delta, beta*s*delta, loc=mu, scale=s*delta).pdf(w_t) ;
    w_t = w_t + 1e-16

    # compute pi_t with numba acceleration
    pi_t = _t(kbar2, T, A, g_m, w_t)

    return(pi_t)


@jit(nopython=True)
def _t(kbar2, T, A, g_m, w_t):

    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    for t in range(T):

        piA = np.dot(pi_mat[t,:],A)
        C = (w_t[t,:]*piA)
        ft = np.sum(C)
        if abs(ft-0) <= 1e-05:
            pi_mat[t+1,1] = 1
        else:
            pi_mat[t+1,:] = C/ft

    pi_t = pi_mat[-1,:]

    return(pi_t)


class memoize(dict):
    """use memoize decorator to speed up compute of the
       transition probability matrix A
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result

@memoize
def  g_t(kbar, b, gamma_kbar):
    """return A, the transition probability matrix
    """

    # compute gammas
    gamma = np.zeros((kbar,1))
    gamma[0,0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        gamma[i,0] = 1-(1-gamma[0,0])**(b**(i))
    gamma = gamma*0.5
    gamma = np.c_[gamma,gamma]
    gamma[:,0] = 1 - gamma[:,0]

    # transition probabilities
    kbar2 = 2**kbar
    prob = np.ones(kbar2)

    for i in range(kbar2):
        for m in range(kbar):
            tmp = np.unpackbits(
                        np.arange(i,i+1,dtype = np.uint16).view(np.uint8))
            tmp = np.append(tmp[8:],tmp[:8])
            prob[i] =prob[i] * gamma[kbar-m-1,tmp[-(m+1)]]

    A = np.fromfunction(
        lambda i,j: prob[np.bitwise_xor(i,j)],(kbar2,kbar2),dtype = np.uint16)

    return(A)


def j_b(x, num_bits):
    """vectorize first part of computing transition probability matrix A
    """

    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2**np.arange(num_bits).reshape([1, num_bits])

    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


@jit(nopython=True)
def s_p(kbar, m0):
    """speed up computation of switching probabilities with Numba
    """

    # switching probabilities
    m1 = 2-m0
    kbar2 = 2**kbar
    g_m = np.zeros(kbar2)
    g_m1 = np.arange(kbar2)

    for i in range(kbar2):
        g = 1
        for j in range(kbar):
            if np.bitwise_and(g_m1[i],(2**j))!=0:
                g = g*m1
            else:
                g = g*m0
        g_m[i] = g

    return(np.sqrt(g_m))


def g_LLb_h(theta, kbar, data):
    """bridge global minimization to local minimization
    """

    theta_in = unpack(theta)
    m0 = theta[1]
    LL = g_LL(m0, kbar, data, theta_in)

    return(LL)


def unpack(theta):
    """unpack theta, package theta_in
    """
    b = theta[0]
    m0 = theta[1]
    gamma_kbar = theta[2]
    delta = theta[3]
    alpha = theta[4]
    mu = theta[5]
    beta = theta[6]
    
    theta_in = [b, gamma_kbar, delta, alpha, mu, beta]

    return(theta_in)

def trajectoires(N,T, data):
    '''
    N nb de trajectoires, 
    T durée sélectionnée pour l'horizon, 
    data = "AAPL" par exemple
    '''
    
    kbar = 5 
    '''
    paramètre du modèle MSM à choisir. correspond au nombres de cycles. Doit être inférieur ou 
    égale à 5 pour des raisons de temps de calculs.
    '''
    Forcast = []
    
    # Formatage des données
    ticker_symbol = data  # Symbole boursier
    start_date = "2022-04-07" # à modifier si besoin
    end_date = "2024-01-08"
    # Récupérer les données historiques
    data = yf.download(ticker_symbol, start=start_date, end=end_date, interval = "60m")
    data['Returns'] = data['Adj Close'].pct_change()
    returns_numpy = data['Returns'].to_numpy()[1:]
    returns_numpy = returns_numpy.reshape(-1,1)
    #print(len(returns_numpy))

    
    # Fitting du modèle
    niter = 1
    temperature = 1.0
    stepsize = 1.0
    parameters, LL, niter, output = glo_min(kbar, returns_numpy, niter, temperature, stepsize)
    # parameters contient les valeurs estimées
    # name parameters for later use:
    b_sim = parameters[0]
    m_0_sim = parameters[1]
    gamma_kbar_sim = parameters[2]
    delta_sim = parameters[3]
    alpha_sim = parameters[4]
    mu_sim = parameters[5]
    beta_sim = parameters[6]
    LL_sim = LL

    gamma_sim = (alpha_sim**2-beta_sim**2)**(1/2)
    mu_sim = -beta_sim*delta_sim/gamma_sim
    tab = []
    gamma_sim = (alpha_sim**2-beta_sim**2)**(1/2)
    mu_sim = -beta_sim*delta_sim/gamma_sim
    for _ in range (N):
        # Valeurs des returns simulés
        returns  = simulatedatanig(b_sim,m_0_sim,gamma_kbar_sim,delta_sim, beta_sim, mu_sim, alpha_sim, kbar, T).reshape(-1)
        # return défini comme ln(p_t/p_t-1)
        prix = [data['Close'].values[-1]]
        for i in range (T):
            p = np.exp(returns[i])
            prix.append(prix[-1]*p*np.exp(-r*1/365))
        Forcast.append(prix)
    return np.array(Forcast)

def afficher (trajectoires):
    plt.figure(figsize = (20,12))
    T = len(trajectoires[1])
    for i in range (len(trajectoires)):
        Time = np.arange(0, T, 1)
        plt.plot(Time, trajectoires[i])
    plt.xlabel("Time")
    plt.ylabel("Prix")
    plt.title("Simulation d'évolution des prix de l'actif sous jacent")
    plt.show()
    
'''
    N nb de trajectoires, 
    T durée sélectionnée pour l'horizon, 
    data = "AAPL" par exemple
'''
    
kbar = 5 
'''
    paramètre du modèle MSM à choisir. correspond au nombres de cycles. Doit être inférieur ou 
    égale à 5 pour des raisons de temps de calculs.
'''

data = "^SPX"
    # Formatage des données
ticker_symbol = data  # Symbole boursier
start_date = "2022-09-07" # à modifier si besoin
end_date = "2024-01-08"
    # Récupérer les données historiques
data = yf.download(ticker_symbol, start=start_date, end=end_date, interval = "60m")
data['Returns'] = data['Adj Close'].pct_change()
returns_numpy = data['Returns'].to_numpy()[1:]
returns_numpy = returns_numpy.reshape(-1,1)
b_sim = 26.08827
m_0_sim = 1.42899
gamma_kbar_sim = 0.0220
delta_sim = 0.0017250  #0.0022
alpha_sim = 280.05081
mu_sim = 0.00051
beta_sim = -0.137944
gamma_sim = (alpha_sim**2-beta_sim**2)**(1/2)
mu_sim = -beta_sim*delta_sim/gamma_sim
tab = []
gamma_sim = (alpha_sim**2-beta_sim**2)**(1/2)
mu_sim = -beta_sim*delta_sim/gamma_sim
print(mu_sim)
T = 365*2+20

for _ in tqdm(range(1001)):
    # Valeurs des returns simulés
    tab.append(simulatedatanig(b_sim,m_0_sim,gamma_kbar_sim,delta_sim, beta_sim, mu_sim, alpha_sim, kbar, T*9).reshape(-1))
# 8h dans la journée
def trajectoires_synth(S0, N,T, data):
    Forcast = []
    for i in range(N):
        # Valeurs des returns simulés
        returns  = tab[i]

        # return défini comme ln(p_t/p_t-1)
        #prix = [data['Close'].values[-1]]
        prix = [S0]

        for j in range (T):
            p = np.exp(returns[9*j])
            prix.append(prix[-1]*p)

        Forcast.append(prix)

        
    return np.array(Forcast)
#print(np.exp(-r))
N=20
data = 0
S0= 4100
T= 360
#tr = trajectoires_synth(S0, N,T, data)



r = 0.025
q = 0.05
    
S = 100
K = 80
sig = 0.3
T = 2

def tridiag_solver(l, d, u, b):
    n = len(b)
    D = np.copy(d)
    B = np.copy(b)
    x = np.zeros(n)
    for i in range(1, n):
        w = l[i] / D[i-1]
        D[i] = D[i] - w*u[i-1]
        B[i] = B[i] - w*B[i-1]
    x[n-1] = B[n-1]/D[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (B[i] - u[i]*x[i+1])/D[i]
    return x

st = time()

# 0,1,2,...,N,N+1

# hyperparamters 
sMin = 10
sMax = 610

N = 4000
M = 365*2

dS = (sMax - sMin)/N
dT = T/M

s = np.zeros(N-1)
tau = np.zeros(M)

l = np.zeros(N-1)
u = np.zeros(N-1)
d = np.zeros(N-1)

vCall = np.zeros(N-1)
vPut = np.zeros(N-1)

alpha = 0.5*sig**2*dT/dS**2
beta = (r - q)*dT/(2.0*dS)

for i1 in range(N-1):
    
    s[i1] = sMin + (i1 + 1)*dS
    if i1 == 0: 
        d[i1]= 1 + r*dT + 2*beta*s[i1]
        u[i1] = -2*beta*s[i1]
    elif i1 == N-2: 
        l[i1] =  2*beta*s[i1]
        d[i1] = 1 + r*dT - 2*beta*s[i1]
    else:
        l[i1] = -alpha*s[i1]**2 + beta*s[i1]
        d[i1] = 1 + r*dT + 2*alpha*s[i1]**2
        u[i1] = -alpha*s[i1]**2 - beta*s[i1]
        
    vCall[i1] = np.maximum(s[i1] - K, 0)
    vPut[i1]  = np.maximum(K - s[i1], 0)
    
# exercise boundaries
SxCall = np.zeros((M,1))
SxPut = np.zeros((M,1))

for j1 in range(M):
    tau[j1] = (j1 + 1)*dT
    vCall = tridiag_solver(l, d, u, vCall)
    flagC = 0
    for i1 in range(N-2, -1, -1):
        # exercise boundary
        if flagC == 0 and vCall[i1] > np.maximum(s[i1] - K, 0):
            SxCall[j1] = s[i1]
            flagC = 1
        # premium
        if vCall[i1] <= np.maximum(s[i1] - K, 0):
            vCall[i1] = np.maximum(s[i1] - K, 0)
    vPut = tridiag_solver(l, d, u, vPut)
    flagP = 0
    for i1 in range(0, N-1):
        # exercise boundary
        if flagP == 0 and vPut[i1] > np.maximum(K - s[i1], 0):
            SxPut[j1] = s[i1]
            flagP = 1
        # premium
        if vPut[i1] <= np.maximum(K - s[i1], 0):
            vPut[i1] = np.maximum(K - s[i1], 0)
            
et = time()
#print('Elapsed time was %f seconds' % (et-st))




eCall = BMS_price('call', s, K, r, q, sig, T)
ePut = BMS_price('put', s, K, r, q, sig, T)
'''
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
'''
# calendar time
tmp = np.append(K,SxCall)
tmp = tmp.reshape(M+1, 1)
#tmp = np.sort(tmp)
tmp = tmp[::-1]
'''
plt.plot(np.append(0, tau), tmp)
plt.xlabel('t')
plt.ylabel('$S^{*}_{t}$')
plt.title('Optimal exercise boundary (call)')
plt.tight_layout()
'''

exerciseBoundary_file = open("exerciseBoundaryCall.dat", "w")
for eb in tmp:
    np.savetxt(exerciseBoundary_file, eb)
exerciseBoundary_file.close()
'''
plt.subplot(2, 2, 2)
plt.plot(s, np.maximum(s - K,0))
plt.plot(s, vCall)
plt.plot(s, eCall);
plt.xlabel('$S_t$')
plt.ylabel('$C_t$')
plt.title('Call')
plt.legend(['Payoff', 'American', 'European'])
plt.tight_layout()

plt.subplot(2, 2, 3)

'''
# calendar time
tmp = np.append(K,SxPut)
tmp = tmp.reshape(M+1, 1)
#tmp = np.sort(tmp)
tmp = tmp[::-1]
'''
plt.plot(np.append(0, tau), tmp)
plt.xlabel('t')
plt.ylabel('$S^{*}_{t}$')
plt.title('Optimal exercise boundary (put)')
plt.tight_layout()
'''
exerciseBoundary_file = open("exerciseBoundaryPut.dat", "w")
tmp.reshape(M+1, 1)
for eb in tmp:
    np.savetxt(exerciseBoundary_file, eb)
exerciseBoundary_file.close()
'''
plt.subplot(2, 2, 4)
plt.plot(s, np.maximum(K - s,0))
plt.plot(s, vPut)
plt.plot(s, ePut);
plt.xlabel('$S_t$')
plt.ylabel('$P_t$')
plt.title('Put')
plt.legend(['Payoff', 'American', 'European'])
plt.tight_layout()

plt.show()
'''

exerciseBoundary = np.loadtxt('exerciseBoundaryPut.dat')
np.random.seed(456718)

numPaths = 1000  # ICIICICICICIICIIIIIIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHI

m = 365 # longueur en nombre de points de la trajectoire
tau = T
s0 = S
dt = tau/m
t = np.linspace(0, 1, m+1) * tau

r = 0.0485
S0 = 4104.8301
ticker_symbol = "^SPX" 
#print(S0)
q = 0.0331026


#set model= 'GBM' ou set = 'mon model'

def pricer_nice_output(s0, numPaths, tau, K , dt, m, r, q, sig, model= 'GBM'):

    s = np.zeros((m+1, numPaths))
    payoffs       = np.zeros(numPaths)
    payoffs_early = np.zeros(numPaths)
    indicators    = np.zeros(numPaths, dtype=int)

    europeanPremium = BMS_price('put', s0, K, r, q, sig, tau)
    st = time()

    if model== 'mon model':
            s = trajectoires_synth(s0, numPaths, m, data)
            print(len(s))
            #print(s)
            print(np.shape(np.array(s)))
            #print(np.array(s).T)
            s = np.array(s).T
    
    for j in range(numPaths):  

        s[0][j]  = s0 
        T = tau 

        # reset the flag
        flag = 0
        
        
        for i in range(1, m+1):
            z = np.random.randn()
            if model=='GBM':
                s[i,j] = s[i-1,j] * np.exp( (r-q-sig*sig/2)*dt + sig*np.sqrt(dt)*z)
            ### ICI ATTENTION l'idee c'est de faire en sorte que votre modele serve pour generer chaque point i 
            ## de la trqjectoire numero j 
            ## truc a faire laisser cette valeur simulee avec la GBM pour voir comment votre simulation se 
            ## se comporte vs ca plus versus Black-schole close price qui est calcule pour l'option europeene car 
            ## il existe une close form solution pour pricer une europeenne option
            ## amuse toi avec les chrono eventuellement mais surtout essaie de prouver que votre modele converge en 
            ## moins de simulations que GBM vers le prix de l'europeenne qui est issu de Black scholes
            ## deja ca serait style


            #print(i,exerciseBoundary[i])

            if flag == 0 and s[i][j] < exerciseBoundary[i]:

                # exercise & discount according to the time of exercise
                # payoffs_early[j] = np.exp(-r*tau)*(K - s[i,j])
                payoffs_early[j] = np.exp(-r*i*dt)*(K - s[i][j])

                # turn the flag on
                flag = 1 

            if i == m:
                payoffs[j] = np.exp(-r*tau) * np.maximum(K - s[i][j], 0)

        if flag == 0:
            payoffs_early[j] = payoffs[j]
            indicators[j] = 1


    #print(len(s))
    #print(s)
    #print(np.shape(np.array(s)))
    
    et = time()
    print(60*'=')
    print('Number of simulations: %i' % numPaths)
    print('Elapsed time was %f seconds.' % (et-st))

    # simulation output
    european = np.mean(payoffs)
    american = np.mean(payoffs_early)

    print('European price (BSM): %f' % europeanPremium)
    print('European price (Sim): %f' % european)
    print('American price (Sim): %f' % american)
    print(60*'=')
    
#pricer_nice_output(S0, numPaths, tau, K, dt, m, r, q, sig, model= 'GBM')
#pricer_nice_output(S0, numPaths, tau, K, dt, m, r, q, sig, model= 'mon model')
def pricer_ed(s0, numPaths, tau, K, dt, m):

    s = np.zeros((m+1, numPaths))
    payoffs       = np.zeros(numPaths)
    payoffs_early = np.zeros(numPaths)
    indicators    = np.zeros(numPaths, dtype=int)

    #europeanPremium = BMS_price('put', s0, K, r, q, sig, tau)
    st = time()

    s = trajectoires_synth(s0, numPaths, m, data)
    #print(len(s))
    #print(s)
    #print(np.shape(np.array(s)))
    #print(np.array(s).T)
    s = np.array(s).T
    
    for j in range(numPaths):  
        #print(s[0][j])
        s[0][j]  = s0 
        
        T = tau 

        # reset the flag
        flag = 0
        
        
        for i in range(1, m+1):
            z = np.random.randn()
            ### ICI ATTENTION l'idee c'est de faire en sorte que votre modele serve pour generer chaque point i 
            ## de la trqjectoire numero j 
            ## truc a faire laisser cette valeur simulee avec la GBM pour voir comment votre simulation se 
            ## se comporte vs ca plus versus Blackschole close price qui est calcule pour l'option europeene car 
            ## il existe une close form solution pour pricer une europeenne option
            ## amuse toi avec les chrono eventuellement mais surtout essaie de prouver que votre modele converge en 
            ## moins de simulations que GBM vers le prix de l'europeenne qui est issu de Black scholes
            ## deja ca serait style


            #print(i,exerciseBoundary[i])

            if flag == 0 and s[i][j] < exerciseBoundary[i]:

                # exercise & discount according to the time of exercise
                # payoffs_early[j] = np.exp(-r*tau)*(K - s[i,j])
                payoffs_early[j] = np.exp(-r*i*dt)*(K - s[i][j])
                # payoffs_early[j] = (K - s[i][j])

                # turn the flag on
                flag = 1 

            if i == m:
                payoffs[j] = np.exp(-r*tau) * np.maximum(s[i][j]-K, 0)
                # payoffs[j] = np.maximum(K - s[i][j], 0)

        if flag == 0:
            payoffs_early[j] = payoffs[j]
            indicators[j] = 1


    #print(len(s))
    #print(s)
    #print(np.shape(np.array(s)))
    
    #et = time()
    #print(60*'=')
    #print('Number of simulations: %i' % numPaths)
    #print('Elapsed time was %f seconds.' % (et-st))

    # simulation output
    european = np.mean(payoffs)
    american = np.mean(payoffs_early)

    #print('European price (BSM): %f' % europeanPremium)
    #print('European price (Sim): %f' % european)
    #print('American price (Sim): %f' % american)
    #print(60*'=')
    
    return european

mu = 2*r
#mu = 0.06
def pricer_ed_2(s0, numPaths, tau, K, dt, m): # maturities 
    
    payoffs       = np.zeros(numPaths)
    payoffs_early = np.zeros(numPaths)
    indicators    = np.zeros(numPaths, dtype=int)
    data = "^SPX"
    #europeanPremium = BMS_price('put', s0, K, r, q, sig, tau)
    st = time()
    s = trajectoires_synth(s0, numPaths, 700, data)
    # simulation d'une trajectoir de longueur 700 jours
    for j in range(numPaths):
        payoffs[j] = np.maximum(s[j][m]*np.exp(mu*m/365)-K, 0)*np.exp(-r*m/365)
        #payoffs[j] =  np.maximum(K-s[j][m], 0)
    return np.mean(payoffs)

#pricer_ed_2(S0, numPaths, tau, 4100, dt, 100)

# Set the start date to compute the maturities
date_str = "2023-04-12"
# create a datetime object from the date string
start_date = datetime.strptime(date_str, "%Y-%m-%d")

def visu():
    df_price = pd.read_csv("spx_4.csv", index_col=0)
    #Filtrer les SPX... qui ne sont pas des SPXW...
    mask = df_price['Calls'].str.contains('SPXW', case=False, na=False)
    df_price = df_price[mask]

    #==================================================================================================================

    callPrices = df_price[['Strike']] #['Last Sale','Strike']]
    # Compute the mid-price
    callPrices['Price'] = np.abs(df_price['Bid'].array + df_price['Ask'].array)/2
    # Convert index to datetime
    callPrices.index = pd.to_datetime(callPrices.index)

    # Getting the weights inversely proportional to bid-ask spread
    callPrices['w'] = np.abs(1/(df_price['Bid'].array - df_price['Ask'].array))

    # define a function to compute the difference in days between two dates
    def date_diff(date):
        diff = (date - start_date)
        return diff.days

    # create a new column in the DataFrame that contains the difference in days
    callPrices['Maturity'] = callPrices.index.to_series().apply(date_diff)
    callPrices['Maturity'] = callPrices['Maturity']/252# trading days.../365.25
    #callPrices['Strike'] = np.log(callPrices['Strike'].array)

    callPrices = callPrices[callPrices['Strike']>=S0]

    # drop today
    callPrices = callPrices[callPrices['Maturity']!=0]
    #579 avec zero mat included


    putPrices = df_price[['Strike']] #['Last Sale','Strike']]
    # Compute the mid-price
    putPrices['Price'] = np.abs(df_price['Bid.1'].array + df_price['Ask.1'].array)/2
    # Convert index to datetime
    putPrices.index = pd.to_datetime(putPrices.index)

    # Getting the weights inversely proportional to bid-ask spread
    putPrices['w'] = np.abs(1/(df_price['Bid.1'].array - df_price['Ask.1'].array))

    # define a function to compute the difference in days between two dates
    def date_diff(date):
        diff = (date - start_date)
        return diff.days

    # create a new column in the DataFrame that contains the difference in days
    putPrices['Maturity'] = putPrices.index.to_series().apply(date_diff)
    putPrices['Maturity'] = putPrices['Maturity']/365.25
    #callPrices['Strike'] = np.log(callPrices['Strike'].array)

    putPrices = putPrices[putPrices['Strike']<=S0]

    # drop today
    putPrices = putPrices[putPrices['Maturity']!=0]
    #579 avec zero mat included
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    #==================================================================================================================
    # create 3D scatter plot
    fig = plt.figure(figsize= [15,15])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(callPrices['Strike'], callPrices['Maturity'], callPrices['Price'])#,s=50)
    #ax.plot_surface(callPrices['Strike'], callPrices['Maturity'], callPrices['Price'], cmap=cm.coolwarm)
    ax.view_init(elev=45, azim=230)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Price')

    plt.show()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    # create 3D scatter plot
    fig = plt.figure(figsize= [15,15])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(callPrices['Strike'], callPrices['Maturity'], callPrices['Price'])#,s=50)
    #ax.plot_surface(callPrices['Strike'], callPrices['Maturity'], callPrices['Price'], cmap=cm.coolwarm)
    ax.view_init(elev=0, azim=230)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Price')

    plt.show()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
def price(numPaths):
# 8h dans la journée
    # Set the start date to compute the maturities
    date_str = "2023-04-12"
    # create a datetime object from the date string
    start_date = datetime.strptime(date_str, "%Y-%m-%d")
    df_price = pd.read_csv("spx_4.csv", index_col=0)
    #Filtrer les SPX... qui ne sont pas des SPXW...
    mask = df_price['Calls'].str.contains('SPXW', case=False, na=False)
    df_price = df_price[mask]

    #==================================================================================================================

    callPrices = df_price[['Strike']] #['Last Sale','Strike']]
    # Compute the mid-price
    import numpy as np
    callPrices['Price'] = np.abs(df_price['Bid'].array + df_price['Ask'].array)/2
    # Convert index to datetime
    callPrices.index = pd.to_datetime(callPrices.index)

    # Getting the weights inversely proportional to bid-ask spread
    callPrices['w'] = np.abs(1/(df_price['Bid'].array - df_price['Ask'].array))

    # define a function to compute the difference in days between two dates
    def date_diff(date):
        diff = (date - start_date)
        return diff.days

    # create a new column in the DataFrame that contains the difference in days
    callPrices['Maturity'] = callPrices.index.to_series().apply(date_diff)
    callPrices['Maturity'] = callPrices['Maturity']/252# trading days.../365.25
    #callPrices['Strike'] = np.log(callPrices['Strike'].array)

    callPrices = callPrices[callPrices['Strike']>=S0]

    # drop today
    callPrices = callPrices[callPrices['Maturity']!=0]
    #579 avec zero mat included


    putPrices = df_price[['Strike']] #['Last Sale','Strike']]
    # Compute the mid-price
    putPrices['Price'] = np.abs(df_price['Bid.1'].array + df_price['Ask.1'].array)/2
    # Convert index to datetime
    putPrices.index = pd.to_datetime(putPrices.index)

    # Getting the weights inversely proportional to bid-ask spread
    putPrices['w'] = np.abs(1/(df_price['Bid.1'].array - df_price['Ask.1'].array))

    # define a function to compute the difference in days between two dates
    def date_diff(date):
        diff = (date - start_date)
        return diff.days

    # create a new column in the DataFrame that contains the difference in days
    putPrices['Maturity'] = putPrices.index.to_series().apply(date_diff)
    putPrices['Maturity'] = putPrices['Maturity']/365.25
    #callPrices['Strike'] = np.log(callPrices['Strike'].array)

    putPrices = putPrices[putPrices['Strike']<=S0]

    # drop today
    putPrices = putPrices[putPrices['Maturity']!=0]
    #579 avec zero mat included

    #==================================================================================================================
    # create 3D scatter plot
    '''
    fig = plt.figure(figsize= [15,15])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(callPrices['Strike'], callPrices['Maturity'], callPrices['Price'])#,s=50)
    #ax.plot_surface(callPrices['Strike'], callPrices['Maturity'], callPrices['Price'], cmap=cm.coolwarm)
    ax.view_init(elev=45, azim=230)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Price')
    '''
    '''

    plt.show()
    # create 3D scatter plot
    fig = plt.figure(figsize= [15,15])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(callPrices['Strike'], callPrices['Maturity'], callPrices['Price'])#,s=50)
    #ax.plot_surface(callPrices['Strike'], callPrices['Maturity'], callPrices['Price'], cmap=cm.coolwarm)
    ax.view_init(elev=0, azim=230)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Price')

    plt.show()
    '''
    strikes = pd.Series(callPrices['Strike'].unique()).sort_values().to_list()
    maturities = pd.Series(callPrices['Maturity'].unique()).sort_values().to_list()
    lenK = len(strikes)
    lenT = len(maturities)

    strike_m = []

    #for i in strikes: 
    for j in maturities: 
            #print(i, j)
            strike_m_temp = callPrices[(callPrices['Maturity']==j)]['Strike'].to_list()
            #marketPrices_temp = callPrices[(callPrices['Maturity']==j)]['Price'].to_list()
            
            #marketPrices.append(marketPrices_temp)
            strike_m.append(strike_m_temp)
            
    strikes = set(strike_m[0])
    for i in range(1,len(strike_m)):
        strikes = strikes.intersection(set(strike_m[i]))

    strikes = list(strikes)
    strikes = sorted(strikes)
    #print(strikes)

    marketPrices = np.zeros((len(strikes), len(maturities)))
    w = np.zeros((len(strikes), len(maturities)))
    for j in range(len(maturities)):
        for i in range(len(strikes)):
            #print(maturities[j])
            #print(strikes[i])
            #print(callPrices[(callPrices['Maturity']== maturities[j]) & (callPrices['Strike']==strikes[i])]['Price'][0])
            marketPrices[i,j] = callPrices[(callPrices['Maturity']== maturities[j]) & (callPrices['Strike']==strikes[i])]['Price'][0]
            w[i,j] = callPrices[(callPrices['Maturity']== maturities[j]) & (callPrices['Strike']==strikes[i])]['w'][0]
            
    marketPrices = marketPrices.T
    w = w.T
    #==================================================================================================================


    strikes_p = pd.Series(putPrices['Strike'].unique()).sort_values().to_list()
    maturities_p = pd.Series(putPrices['Maturity'].unique()).sort_values().to_list()
    lenK_p = len(strikes_p)
    lenT_p = len(maturities_p)

    strike_m_p = []

    #for i in strikes: 
    for j in maturities_p: 
            #print(i, j)
            strike_m_temp_p = putPrices[(putPrices['Maturity']==j)]['Strike'].to_list()
            #marketPrices_temp = callPrices[(callPrices['Maturity']==j)]['Price'].to_list()
            
            #marketPrices.append(marketPrices_temp)
            strike_m_p.append(strike_m_temp_p)
            
    strikes_p = set(strike_m_p[0])
    for i in range(1,len(strike_m_p)):
        strikes_p = strikes_p.intersection(set(strike_m_p[i]))

    strikes_p = list(strikes_p)
    strikes_p = sorted(strikes_p)
    #print(strikes_p)
    #print(maturities_p)
    marketPrices_p = np.zeros((len(strikes_p), len(maturities_p)))
    w_p = np.zeros((len(strikes_p), len(maturities_p)))
    for j in range(len(maturities_p)):
        for i in range(len(strikes_p)):
            #print(maturities_p[j])
            #print(strikes_p[i])
            #print(putPrices[(putPrices['Maturity']== maturities_p[j]) & (putPrices['Strike']==strikes_p[i])]['Price'][0])
            marketPrices_p[i,j] = putPrices[(putPrices['Maturity']== maturities_p[j]) & (putPrices['Strike']==strikes_p[i])]['Price'][0]
            w_p[i,j] = putPrices[(putPrices['Maturity']== maturities_p[j]) & (putPrices['Strike']==strikes_p[i])]['w'][0]
            
    marketPrices_p = marketPrices_p.T
    w_p = w_p.T
    # function for the search: 
    def myRange(start, finish, increment):
        while (start <= finish):
            yield start
            start += increment
            
    def objFunc(v, x0, x1, x2):
        # Paraboloid centered on (x, y), with scale factors (10, 20) and minimum 30
        return 10.0*(v[0]-x0)**2 + 20.0*(v[1]-x1)**2 + 30.0*(v[2]-x2)**2 + 40.0

    #params2 = xopt
    lenT = len(maturities)
    lenK = len(strikes)
    modelPrices = np.zeros((lenT, lenK))
    #print

    #print('lenT', lenT)
    #print('lenK', lenK)
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")

    for i in tqdm(range(lenT)):
        for j in range(lenK):
            m = (int(maturities[i]*360))
            K = strikes[j]
            #[km, cT_km] = mfc.genericFFT(params2, S0, K, r, q, T, alpha, eta, n, model)
            #modelPrices[i,j] = cT_km[0]
            modelPrices[i,j] = pricer_ed_2(S0, numPaths, 0, K, dt, m)
            
    # plot
    import plotly.graph_objects as go
    import numpy as np

    # Générer une palette de couleurs allant du bleu au rouge pour les 23 maturités
    colors = [f'rgb({int(255 * i / 23)}, 0, {int(255 * (23 - i) / 23)})' for i in range(23)]

    # Créer la figure
    fig = go.Figure()

    # Ajouter les prix du marché en tant que nuage de points avec légende
    for i in range(len(maturities)):
        fig.add_trace(go.Scatter(x=strikes, y=marketPrices[i,:], mode='markers', name='Market T = ' + str(maturities[i]), marker=dict(color=colors[i])))

    # Ajouter les prix du modèle en tant que nuage de points sans légende
    for i in range(len(maturities)):
        fig.add_trace(go.Scatter(x=strikes, y=modelPrices[i,:], mode='markers', showlegend=False, marker=dict(symbol='cross', color=colors[i])))

    # Mettre à jour la mise en page
    fig.update_layout(
        title='Market vs. Markov NIG Model',
        xaxis_title='Strike',
        yaxis_title='Price',
        legend=dict(
            x=1.05,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=1000,
        width=1000,
        font=dict(size=10)
    )

    # Afficher le graphique
    fig.show()



    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    #print(S0)
    maturities_, strikes_ = np.meshgrid(maturities, strikes)

    maturities_flat = maturities_.flatten()
    strikes_flat =  strikes_.flatten()
    prices_flat =  marketPrices.flatten()

    import plotly.graph_objects as go

    # Création de la figure
    fig = go.Figure()

    # Ajout des données pour le modèle Heston
    fig.add_trace(go.Scatter3d(
        x=strikes_flat,
        y=maturities_flat,
        z=modelPrices.T.flatten(),
        mode='markers',
        marker=dict(size=5),
        name='MSM'
    ))

    # Ajout des données pour les prix du marché
    fig.add_trace(go.Scatter3d(
        x=strikes_flat,
        y=maturities_flat,
        z=marketPrices.T.flatten(),
        mode='markers',
        marker=dict(size=5),
        name='Market'
    ))

    # Mise en forme du tracé
    fig.update_layout(
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Maturity',
            zaxis_title='Price',
            camera=dict(
                eye=dict(x=-1.7, y=-1.7, z=0.5)
            )
        ),
        title='Market vs. Model',
        width=1000,  # Largeur de la figure
        height=1200  # Hauteur de la figure
    )

    # Affichage du tracé
    fig.show()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
