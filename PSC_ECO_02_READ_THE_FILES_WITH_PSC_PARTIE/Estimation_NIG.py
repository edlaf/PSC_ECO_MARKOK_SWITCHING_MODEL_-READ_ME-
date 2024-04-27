import numpy as np
import pandas as pd
import scipy.optimize as opt
from statsmodels import regression
import statsmodels.formula.api as sm
from numba import jit, njit, prange, float64, int64
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import norm, norminvgauss
from tqdm import tqdm
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
# Filtrer les avertissements RuntimeWarning



# NOUS DEFINISSONS PLUSIEURS ESTIMATIONS EN FONCTION DE L'ACTIF CONSIDERE

warnings.filterwarnings("ignore", category=RuntimeWarning)
def glo_min_2(kbar, data, niter, temperature, stepsize):
    """2-step basin-hopping method combines global stepping algorithm
       with local minimization at each step.
    """

    """step 1: local minimizations
    """
    theta = loc_min_2(kbar, data)

    """step 2: global minimum search uses basin-hopping
       (scipy.optimize.basinhopping)
    """
    # objective function
    f = g_LLb_h

    # x0 = initial guess, being theta, from Step 1.
    # Presents as: [b, m0, gamma_kbar, sigma] b,m0,gamma_kbar,delta, alpha
    x0 = theta
    #print("First optimization ------DONE------ :    ", x0)
    # basinhopping arguments
    niter = niter
    T = temperature
    stepsize = stepsize
    args = (kbar, data)

    # bounds
    bounds = ((2,3.0),(1.47,1.58),(0.15,0.3),(0.15,0.23), (1440, 1500), (-1,1), (-8.1,-6.5)) # m_O 1,39 et alpha 60, beta 7 gamma 0.1

    # minimizer_kwargs
    minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds, args = args)

    res = opt.basinhopping(
                func = f, x0 = x0, niter = niter, T = T, stepsize = stepsize,
                minimizer_kwargs = minimizer_kwargs)

    parameters, LL, niter, output = res.x,res.fun,res.nit,res.message
    
    s = sc.skew(data)[0]
    parameters[6] = np.abs(s)/s*np.abs(parameters[6])
    #parameters[6] = parameters[6]*(1/np.sqrt(parameters[1])*1/2+1/2*1/np.sqrt(2-parameters[1]))**(-kbar)
    #parameters[5] = np.abs(s)/parameters[5]
    LL_sim = LL
    #print("Second optimization ------DONE------ :    ", parameters)

    return(parameters, LL, niter, output)

def estim_delta_alpha(data):
    a = np.mean(data**2)
    b = np.mean(data**4)
    alph = (3*a/(b-3*a**2))**(1/2) #alpha 
    delt = (3*a/(b-3*a**2))**(1/2)*a
    return [delt, alph]

def loc_min_2(kbar, data):
    """step 1: local minimization
       parameter estimation uses bounded optimization (scipy.optimize.fminbound)
    """

    # set up
    b = np.array([1.5, 2, 3, 4, 5, 6, 7, 8, 10,11, 12, 15,17,18,19,19.5,20,20.5,21,22, 24, 26, 27, 30, 35])
    b = np.array([20])
    #b = np.linspace(3,4,3) #marche
    b = np.linspace(2.3,2.8,4) # 1
    lb = len(b)
    #gamma_kbar = np.array([0.0050,0.01,0.15,0.02,0.04,0.05,0.06,0.07,0.1,0.15, 0.2, 0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])# Voir s'il faut garder 0.01
    gamma_kbar = np.linspace(0.15,0.25,3) #0.1,0?14 marche
    lg = len(gamma_kbar)

    beta = np.linspace(-10,-2,3)
    beta= [-7]# -8,-3 marche
    lbeta = len(beta)
    est =  estim_delta_alpha(data)
    delta = est[0]
    delta = 0.2
    alpha = est[1]
    #alpha = np.linspace(50,60,4) #marche
    alpha = [1480.0]
    lalpha = len(alpha)
    s = sc.skew(data)[0]
    #beta = np.abs(s)/s*s/3*alpha**(3/2)*delta**(1/2)/3
    #print(beta)
    #gamma = (alpha**2-beta**2)**(1/2)
    mu = 0#-beta*delta/gamma
    #print(mu)
    #a = np.mean(data**2)
    #b = np.mean(data**4)
    #alpha = (3*a/(b-3*a**2))**(1/2) #alpha 
    #delta = (3*a/(b-3*a**2))**(1/2)*a # delta

    # templates
    theta_out = np.zeros(((lb*lg*lbeta*lalpha),5))
    theta_LLs = np.zeros((lb*lg*lbeta*lalpha))

    # objective function
    f = g_LL

    # bounds
    m0_l = 1.45
    m0_u = 1.6
    b_sim = 26.08827 
    m_0_sim = 1.42899 
    gamma_kbar_sim = 0.0220 
    delta_sim = 0.001250 
    alpha_sim = 280.05081 
    mu_sim = 0.00051 
    beta_sim = -0.137944

    # Optimizaton stops when change in x between iterations is less than xtol
    xtol = 1e-05

    # display: 0, no message; 1, non-convergence; 2, convergence;
    # 3, iteration results.
    disp = 1

    idx = 0

                
    
    idx = np.argsort(theta_LLs)

    theta_LLs = np.sort(theta_LLs)
    theta = [2.3, 1.5072949016875157, 0.15, 0.2, 1480.0, 0, -7.0]
    theta_out = theta_out[idx,:]
    #print(theta)
    return theta

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
        dat[k] = np.sqrt(np.prod(tmp))
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
    bounds = ((0.001,50),(1,1.554),(0.0001,0.2),(1e-5,5), (10, 400), (-1,1), (-4.43,-1)) # m_O 1,39 et alpha 60, beta 7 gamma 0.1

    # minimizer_kwargs
    minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds, args = args)

    res = opt.basinhopping(
                func = f, x0 = x0, niter = niter, T = T, stepsize = stepsize,
                minimizer_kwargs = minimizer_kwargs)

    parameters, LL, niter, output = res.x,res.fun,res.nit,res.message
    
    s = sc.skew(data)[0]
    parameters[6] = np.abs(s)/s*np.abs(parameters[6])
    #parameters[6] = parameters[6]*(1/np.sqrt(parameters[1])*1/2+1/2*1/np.sqrt(2-parameters[1]))**(-kbar)
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
    b = np.array([1.5, 2, 3, 4, 5, 6, 7, 8, 10,11, 12, 15,17,18,19,19.5,20,20.5,21,22, 24, 26, 27, 30, 35])
    b = np.array([20])
    #b = np.linspace(3,4,3) #marche
    b = np.linspace(0.01,1,5) 
    lb = len(b)
    #gamma_kbar = np.array([0.0050,0.01,0.15,0.02,0.04,0.05,0.06,0.07,0.1,0.15, 0.2, 0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])# Voir s'il faut garder 0.01
    gamma_kbar = np.linspace(0.0001,0.64,5) #0.1,0?14 marche
    lg = len(gamma_kbar)

    beta = np.linspace(-4,0,5)
    beta = [-3]# -8,-3 marche
    lbeta = len(beta)
    est =  estim_delta_alpha(data)
    delta = est[0]
    delta = 0.02
    alpha = est[1]
    #alpha = np.linspace(50,60,4) #marche
    alpha = [30,50,60,70]
    alpha = [50,60,70]
    lalpha = len(alpha)
    s = sc.skew(data)[0]
    #beta = np.abs(s)/s*s/3*alpha**(3/2)*delta**(1/2)/3
    #print(beta)
    #gamma = (alpha**2-beta**2)**(1/2)
    mu = 0#-beta*delta/gamma
    #print(mu)
    #a = np.mean(data**2)
    #b = np.mean(data**4)
    #alpha = (3*a/(b-3*a**2))**(1/2) #alpha 
    #delta = (3*a/(b-3*a**2))**(1/2)*a # delta

    # templates
    theta_out = np.zeros(((lb*lg*lbeta*lalpha),5))
    theta_LLs = np.zeros((lb*lg*lbeta*lalpha))

    # objective function
    f = g_LL

    # bounds
    m0_l = 1.05
    m0_u = 1.3
    b_sim = 26.08827 
    m_0_sim = 1.42899 
    gamma_kbar_sim = 0.0220 
    delta_sim = 0.001250 
    alpha_sim = 280.05081 
    mu_sim = 0.00051 
    beta_sim = -0.137944

    # Optimizaton stops when change in x between iterations is less than xtol
    xtol = 1e-05

    # display: 0, no message; 1, non-convergence; 2, convergence;
    # 3, iteration results.
    disp = 1

    idx = 0
    for i in tqdm(range(lb)):
        for j in range(lg):
            for k in range(lbeta):
                for l in range(lalpha):

            # args
                    theta_in = [b[i], gamma_kbar[j], delta, alpha[l], mu, beta[k]]
                    args = (kbar, data, theta_in)

                    xopt, fval, ierr, numfunc = opt.fminbound(
                                func = f, x1 = m0_l, x2 = m0_u, xtol = xtol,
                                args = args, full_output = True, disp = disp)

                    m0, LL = xopt, fval
                    theta_out[idx,:] = b[i], m0, gamma_kbar[j], beta[k], alpha[l]

                    theta_LLs[idx] = LL
                    idx +=1
                

    idx = np.argsort(theta_LLs)

    theta_LLs = np.sort(theta_LLs)
    print(theta_LLs[0])

    theta = theta_out[idx[0],:].tolist()[:3]+[delta,theta_out[idx[0],:].tolist()[-1], mu]+[theta_out[idx[0],:].tolist()[-2]]
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
    LL = _LL(kbar2, T, A, g_m, w_t)#+(mu-beta*delta/gamma)**4+(beta - sk/3*alpha**(3/2)*delta**(1/2)/3)**4

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

def visu_NIG():
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
# Télécharger les données historiques pour une action spécifique
    ticker_symbol = "^SPX"  # Symbole boursier d'Apple Inc. (AAPL)
    start_date = "2000-01-01"
    end_date = "2024-01-01"

    # Récupérer les données historiques
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Calculer les rendements
    data['Returns'] = data['Adj Close'].pct_change()

    # Calculer les rendements
    data['Returns'] = data['Adj Close'].pct_change()

    # Tracer les rendements en fonction du temps

    # Extraire les rendements sous forme de tableau NumPy
    returns_numpy = data['Returns'].to_numpy()[1:]
    #returns_numpy = (returns_numpy-np.mean(returns_numpy))/np.std(returns_numpy)
    print("Nombre de dates prises en compte :", len(returns_numpy))
    print(data)
    returns_numpy = returns_numpy.reshape(-1,1)
    kbar = 5
    niter = 5
    temperature = 0.5
    stepsize = 0.5
    T = len(returns_numpy)+1
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    parameters, LL, niter, output = glo_min(kbar, returns_numpy, niter, temperature, stepsize)

    # parameters contient les valeurs estimées
    # name parameters for later use:
    # name parameters for later use:
    b_sim = parameters[0]
    m_0_sim = parameters[1]
    gamma_kbar_sim = parameters[2]
    delta_sim = parameters[3]
    alpha_sim = parameters[4]
    mu_sim = parameters[5]
    beta_sim = parameters[6]
    LL_sim = LL
    

    data_sim = simulatedatanig(b_sim,m_0_sim,gamma_kbar_sim,delta_sim, beta_sim, mu_sim, alpha_sim, kbar, T-1).reshape(-1)
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")

    #print("Valeurs mises au depart :", "b =", b,"m_0=", m0, "gamma_kbar =",gamma_kbar,"delta =", delta, "alpha =",alpha, "beta =", beta, "mu =", mu)
    print("Parameters from glo_min for Simulated dataset: ", "\n"
        "kbar = ", kbar,"\n"
        'b = %.5f' % b_sim,"\n"
        'm_0 = %.5f' % m_0_sim,"\n"
        'gamma_kbar = %.5f' % gamma_kbar_sim,"\n"
        'delta = %.5f' % (delta_sim),"\n"
        'alpha = %.5f' % (alpha_sim),"\n"
        'mu = %.5f' % (mu_sim),"\n"
        'beta = %.5f' % (beta_sim),"\n"
        'Likelihood = %.5f' % LL_sim,"\n"
        "niter = " , niter,"\n"
        "output = " , output,"\n")

    T = len(returns_numpy)+1
    import plotly.graph_objects as go
    data_sim = simulatedatanig(b_sim,m_0_sim,gamma_kbar_sim,delta_sim, beta_sim, mu_sim, alpha_sim, kbar, T).reshape(-1)
    s = sc.skew(returns_numpy)[0]
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data_sim, mode='lines', name='Returns simulés MSM avec loi Normal', line=dict(color='rgba(255, 0, 0, 0.75)')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Returns'], mode='lines', name='Returns réels', line=dict(color='rgba(0, 0, 255, 0.75)')))

    fig.update_layout(
        title='Rendements simulés en fonction du temps (' + ticker_symbol + ')',
        xaxis_title='Date',
        yaxis_title='Rendements',
        legend=dict(x=0.01, y=0.99),
        height=600,
        width=1000,
        font=dict(size=20)
    )

    fig.show()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")

    import plotly.graph_objects as go

    fig = go.Figure()


    fig.add_trace(go.Histogram(x=data['Returns'], nbinsx=200, xbins=dict(size=0.0005), histnorm='probability', name='Returns réels', opacity=0.7))
    fig.add_trace(go.Histogram(x=data_sim, nbinsx=200, xbins=dict(size=0.0005), histnorm='probability', name='Returns simulés', opacity=0.7))

    fig.update_layout(
        title='Historigramme comparatif des returns réels et simulés (' + ticker_symbol + ')',
        xaxis_title='Valeur des returns',
        yaxis_title='Répartition des returns',
        legend=dict(x=0.01, y=0.99),
        height=600,
        width=1000,
        bargap=0,
        font=dict(size=20)
    )

    fig.show()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    data_sim = simulatedatanig(b_sim,m_0_sim,gamma_kbar_sim,delta_sim, beta_sim, mu_sim, alpha_sim, kbar, T).reshape(-1)
    s = sc.skew(returns_numpy)[0]

    Time = np.arange(0, T, 1)
    plt.figure(figsize = (20,12))
    plt.plot(data.index, data_sim, label ='returns simulés MSM avec loi NIG', alpha =0.5)
    plt.title('Rendements simulés en fonction du temps (' + ticker_symbol + ')')
    plt.plot(data.index, data['Returns'], color='blue', linestyle='-', label = 'returns réels', alpha =0.5)
    plt.xlabel('Date')
    plt.ylabel('Rendements')
    plt.legend()
    plt.plot()

    plt.figure(figsize = (20,12))
    plt.hist(data_sim, bins =200, density = True, alpha =0.5, label ='returns simulés')
    plt.hist(data['Returns'], density = True, bins =200, alpha =0.5, label = 'returns réels')
    plt.title('Historigramme comparatif des returns réels et simulés (' + ticker_symbol + ')')
    plt.ylabel('Répartition des returns')
    plt.xlabel('Valeur des returns')
    plt.legend()
    plt.plot()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")


def analyse_data(tab):
    L = len(tab)
    kbar_ = [1,2,3, 4, 5, 6]
    
    analyse = [[],[],[],[],[]]# b_sim, M_0, gamma, delta, alpha, mu, beta, LL
    index   = ['b', 'm_0', 'gamma', 'sigma','LL']
    columns = ['kbar = 1','kbar = 2','kbar = 3', 'kbar = 4', 'kbar = 5', 'kbar = 6']
    for i in range (len(tab)):
        # Télécharger les données historiques pour une action spécifique
        ticker_symbol = tab[i]  # Symbole boursier d'Apple Inc. (AAPL)
        start_date = "2000-01-01"
        end_date = "2024-01-01"

        # Récupérer les données historiques
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        # Calculer les rendements
        data['Returns'] = data['Adj Close'].pct_change()
        # Extraire les rendements sous forme de tableau NumPy
        returns_numpy = data['Returns'].to_numpy()[1:]
        returns_numpy = returns_numpy.reshape(-1,1)
        for j in range (len(kbar_)):
            kbar__ = kbar_[j]
            
            niter = 5
            temperature = 0.5
            stepsize = 0.5
            parameters, LL, niter, output = glo_min(kbar__, returns_numpy, niter, temperature, stepsize)
            # parameters contient les valeurs estimées
            # name parameters for later use:
            # name parameters for later use:
            analyse[0].append(parameters[0]) 
            analyse[1].append(parameters[1]) 
            analyse[2].append(parameters[2]) 
            analyse[3].append(parameters[3]) 
            analyse[4].append(LL)
    df = pd.DataFrame(analyse, columns=columns, index=index)
    return df.transpose()

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*encountered.*NaN.*")
def convergence (N_2, kbar_max):
    #N_s = np.geomspace(1, N, num=nb_inter, dtype=int)
    ticker_symbol = "^SPX"  # Symbole boursier d'Apple Inc. (AAPL)
    start_date = "1900-01-01"
    end_date = "2024-01-01"
    # Récupérer les données historiques
    data               = yf.download(ticker_symbol, start=start_date, end=end_date)
    returns_numpy = data['Close'].to_numpy()[1:]
    for i in range(len(returns_numpy)-1):
        returns_numpy[i] = np.log(returns_numpy[i+1]/returns_numpy[i])
    returns_numpy[-1] = 0
    returns_numpy = np.concatenate((returns_numpy, [0]))
    returns_numpy = returns_numpy.reshape(-1,1)

    '''
    for i in tqdm(range (len(N_s))):
        forcast, returns, MAE, MSE, R = residus_1(name, horizon, kbar, N_s[i])
        MAE_s.append(MAE)
        MSE_s.append(MSE)
        R_s.append(R)
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Convergence des résidus MAE', 'Convergence des résidus MSE', 'Convergence des résidus R'))
    # Add traces
    fig.add_trace(go.Scatter(x=N_s, y=MAE_s, mode='lines', name='MAE'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_s, y=MSE_s, mode='lines', name='MSE'), row=1, col=2)
    fig.add_trace(go.Scatter(x=N_s, y=R_s, mode='lines', name='R'), row=2, col=1)

    # Update layout
    fig.update_layout(height=600, width=1000, title_text="Convergence des résidus")
    fig.update_xaxes(type="log")
    #fig.update_yaxes(type="log")
    # Show plot
    fig.show()
    '''
    # moments
    moments_1 = []
    moments_2 = []
    moments_3 = []
    # Fitting du modèle
    niter          = 8
    temperature    = 0.2
    stepsize       = 0.2
    parameters, LL, niter, output = glo_min_2(kbar_max, returns_numpy, niter, temperature, stepsize)
    print( "-----------------------------------fitting done-----------------------------------" )
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
    k = 0
    v = 0
    m = 0
    T = len(returns_numpy)+1
    for i in tqdm(range (N_2)):
        data_sim = simulatedatanig(b_sim,m_0_sim,gamma_kbar_sim,delta_sim, beta_sim, mu_sim, alpha_sim, kbar_max, T-1).reshape(-1)
        k+= sc.skew(data_sim)/100
        v += np.std(data_sim)/100
        m +=np.mean(data_sim)/100
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    print("Entrainement sur les données depuis 1930")
    print("Moment d'ordre 1 réel :", np.mean(returns_numpy))
    print("Moment d'ordre 1 simulé moyenné sur ",N_2,"trajectoires:", m)
    print("L'erreur relative est de :", np.abs((np.mean(returns_numpy)-m)/m))
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    print("Moment d'ordre 1 réel :", np.std(returns_numpy))
    print("Moment d'ordre 2 simulé moyenné sur ",N_2,"trajectoires:", v)
    print("L'erreur relative est de :", np.abs((np.std(returns_numpy)-v)/np.std(returns_numpy)))
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    print("Moment d'ordre 3 réel :", sc.skew(returns_numpy)[0])
    print("Moment d'ordre 3 simulé moyenné sur ",N_2,"trajectoires:", k)
    print("L'erreur relative est de :", np.abs((sc.skew(returns_numpy)-k)/sc.skew(returns_numpy)))
    return
warnings.filterwarnings("ignore", message="NaN result encountered.")

# simualtion des données MSM
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

def glo_min_3(kbar, data, niter, temperature, stepsize):
    """2-step basin-hopping method combines global stepping algorithm
       with local minimization at each step.
    """

    """step 1: local minimizations
    """
    theta, theta_LLs, theta_out, ierr, numfunc = loc_min_3(kbar, data)

    """step 2: global minimum search uses basin-hopping
       (scipy.optimize.basinhopping)
    """
    # objective function
    f = g_LLb_h_3

    # x0 = initial guess, being theta, from Step 1.
    # Presents as: [b, m0, gamma_kbar, sigma] b,m0,gamma_kbar,delta, alpha
    x0 = theta
    #print("First optimization ------DONE------ :    ", x0)
    # basinhopping arguments
    niter = niter
    T = temperature
    stepsize = stepsize
    args = (kbar, data)

    # bounds
    bounds = ((1,50),(1,1.99),(0.000001,0.4),(1e-5,5), (7, 100), (-1,1), (-6,0)) # m_O 1,39 et alpha 60, beta 7 gamma 0.1

    # minimizer_kwargs
    minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds, args = args)

    res = opt.basinhopping(
                func = f, x0 = x0, niter = niter, T = T, stepsize = stepsize,
                minimizer_kwargs = minimizer_kwargs)

    parameters, LL, niter, output = res.x,res.fun,res.nit,res.message
    
    s = sc.skew(data)[0]
    parameters[6] = np.abs(s)/s*np.abs(parameters[6])
    #parameters[6] = parameters[6]*(1/np.sqrt(parameters[1])*1/2+1/2*1/np.sqrt(2-parameters[1]))**(-kbar)
    #parameters[5] = np.abs(s)/parameters[5]
    LL_sim = LL
    #print("Second optimization ------DONE------ :    ", parameters)

    return(parameters, LL, niter, output)

def estim_delta_alpha(data):
    a = np.mean(data**2)
    b = np.mean(data**4)
    alph = (3*a/(b-3*a**2))**(1/2) #alpha 
    delt = (3*a/(b-3*a**2))**(1/2)*a
    return [delt, alph]

def loc_min_3(kbar, data):
    """step 1: local minimization
       parameter estimation uses bounded optimization (scipy.optimize.fminbound)
    """

    # set up
    b = np.array([1.5, 2, 3, 4, 5, 6, 7, 8, 10,11, 12, 15,17,18,19,19.5,20,20.5,21,22, 24, 26, 27, 30, 35])
    
    #b = np.linspace(3,4,3) #marche
    b = np.array([4,8]) # 1
    lb = len(b)
    #gamma_kbar = np.array([0.0050,0.01,0.15,0.02,0.04,0.05,0.06,0.07,0.1,0.15, 0.2, 0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])# Voir s'il faut garder 0.01
    #gamma_kbar = np.linspace(0.0001,0.64,5) #0.1,0?14 marche
    gamma_kbar = np.linspace(0.001,0.98,5)
    lg = len(gamma_kbar)

    beta = np.linspace(-10,-2,3)
    beta= [-1]# -8,-3 marche
    lbeta = len(beta)
    est =  estim_delta_alpha(data)
    delta = est[0]
    delta = 0.01
    alpha = est[1]
    #alpha = np.linspace(50,60,4) #marche
    alpha = [10,20,30]
    lalpha = len(alpha)
    s = sc.skew(data)[0]
    #beta = np.abs(s)/s*s/3*alpha**(3/2)*delta**(1/2)/3
    #print(beta)
    #gamma = (alpha**2-beta**2)**(1/2)
    mu = 0#-beta*delta/gamma
    #print(mu)
    #a = np.mean(data**2)
    #b = np.mean(data**4)
    #alpha = (3*a/(b-3*a**2))**(1/2) #alpha 
    #delta = (3*a/(b-3*a**2))**(1/2)*a # delta

    # templates
    theta_out = np.zeros(((lb*lg*lbeta*lalpha),5))
    theta_LLs = np.zeros((lb*lg*lbeta*lalpha))

    # objective function
    f = g_LL_3

    # bounds
    m0_l = 1.01
    m0_u = 1.99
    b_sim = 26.08827 
    m_0_sim = 1.42899 
    gamma_kbar_sim = 0.0220 
    delta_sim = 0.001250 
    alpha_sim = 280.05081 
    mu_sim = 0.00051 
    beta_sim = -0.137944

    # Optimizaton stops when change in x between iterations is less than xtol
    xtol = 1e-05

    # display: 0, no message; 1, non-convergence; 2, convergence;
    # 3, iteration results.
    disp = 1

    idx = 0
    for i in range(lb):
        for j in range(lg):
            for k in range(lbeta):
                for l in range(lalpha):

            # args
                    theta_in = [b[i], gamma_kbar[j], delta, alpha[l], mu, beta[k]]
                    args = (kbar, data, theta_in)

                    xopt, fval, ierr, numfunc = opt.fminbound(
                                func = f, x1 = m0_l, x2 = m0_u, xtol = xtol,
                                args = args, full_output = True, disp = disp)

                    m0, LL = xopt, fval
                    theta_out[idx,:] = b[i], m0, gamma_kbar[j], beta[k], alpha[l]

                    theta_LLs[idx] = LL
                    idx +=1
                

    idx = np.argsort(theta_LLs)

    theta_LLs = np.sort(theta_LLs)
    #print(theta_LLs[0])

    theta = theta_out[idx[0],:].tolist()[:3]+[delta,theta_out[idx[0],:].tolist()[-1], mu]+[theta_out[idx[0],:].tolist()[-2]]
    theta_out = theta_out[idx,:]
    #print(theta)
    return(theta, theta_LLs, theta_out, ierr, numfunc)


def g_LL_3(m0, kbar, data, theta_in):
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
    A = g_t_3(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p_1(kbar, m0)

    # volatility model
    s = g_m

    # returns
    w_t = data
    w_t = norminvgauss(alpha*s*delta, beta*s*delta, loc=mu, scale=s*delta).pdf(w_t) ;
    w_t = w_t + 1e-16

    # log likelihood using numba
    sk = sc.skew(data)[0]
    gamma = (np.abs(alpha**2-beta**2))**(1/2)
    LL = _LL_3(kbar2, T, A, g_m, w_t)#+(mu-beta*delta/gamma)**4+(beta - sk/3*alpha**(3/2)*delta**(1/2)/3)**4

    return(LL)


@jit(nopython=True)
def _LL_3(kbar2, T, A, g_m, w_t):
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


def g_pi_t_3(m0, kbar, data, theta_in):
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
    A = g_t_3(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p_1(kbar, m0)

    # volatility model
    s = g_m

    # returns
    w_t = data
    w_t = norminvgauss(alpha*s*delta, beta*s*delta, loc=mu, scale=s*delta).pdf(w_t) ;
    w_t = w_t + 1e-16

    # compute pi_t with numba acceleration
    pi_t = _t_3(kbar2, T, A, g_m, w_t)

    return(pi_t)


@jit(nopython=True)
def _t_3(kbar2, T, A, g_m, w_t):

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
def  g_t_3(kbar, b, gamma_kbar):
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


def j_b_3(x, num_bits):
    """vectorize first part of computing transition probability matrix A
    """

    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2**np.arange(num_bits).reshape([1, num_bits])

    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


@jit(nopython=True)
def s_p_1(kbar, m0):
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

    return(g_m)


def g_LLb_h_3(theta, kbar, data):
    """bridge global minimization to local minimization
    """

    theta_in = unpack_3(theta)
    m0 = theta[1]
    LL = g_LL_3(m0, kbar, data, theta_in)

    return(LL)


def unpack_3(theta):
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

import plotly.graph_objects as go
def conv_2 (N, nb_inter):
    N_s_2 = np.geomspace(100, N, num=nb_inter, dtype=int)
    kbar = 3
    # Fitting du modèle
    niter = 5
    temperature = 0.5
    stepsize = 0.5
    b_sim_s= []
    m_0_sim_s= []
    gamma_kbar_sim_s= []
    delta_sim_s= []
    beta_sim_s = []
    mu_sim_s = []
    alpha_sim_s = []
    LL_sim_s = []
    #b = 0.60918
    b_sim = 7.7
    m_0_sim = 1.67 
    gamma_kbar_sim  = 0.002
    delta_sim = 0.12
    alpha_sim = 14.2
    mu_sim = 0.002
    beta_sim = -3.2
    returns = simulatedatanig(b_sim,m_0_sim,gamma_kbar_sim,delta_sim, beta_sim, mu_sim, alpha_sim, kbar, N_s_2[-1])
    for i in tqdm(range (len(N_s_2))):
        returns_1 = returns[:N_s_2[i]]
        # Fitting du modèle
        niter = 5
        temperature = 0.5
        stepsize = 0.5
        parameters, LL, niter, output = glo_min_3(kbar, returns_1, niter, temperature, stepsize)
        b_sim_s.append(parameters[0])
        m_0_sim_s.append(parameters[1])
        gamma_kbar_sim_s.append(parameters[2])
        delta_sim_s.append(parameters[3])
        alpha_sim_s.append(parameters[4])
        mu_sim_s.append(parameters[5])
        beta_sim_s.append(parameters[6])
        LL_sim_s.append(LL)
    # Création des subplots avec deux graphiques par ligne
    fig = make_subplots(rows=4, cols=2, subplot_titles=('Convergence de m_0', 'Convergence de gamma_k', 'Convergence de delta', 'Convergence de alpha', 'Convergence de mu', 'Convergence de beta', 'Convergence de b', 'Evolution de LL'),
                        row_heights=[2.5, 2.5,2.5, 2.5])

    # Graphique de convergence de m_0
    fig.add_trace(go.Scatter(x=N_s_2, y=m_0_sim_s, mode='lines', name='simulation'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[m_0_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=1, col=1)

    # Graphique de convergence de gamma_k
    fig.add_trace(go.Scatter(x=N_s_2, y=gamma_kbar_sim_s, mode='lines', name='simulation'), row=1, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=[gamma_kbar_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=1, col=2)

    # Graphique de convergence de delta
    fig.add_trace(go.Scatter(x=N_s_2, y=delta_sim_s, mode='lines', name='simulation'), row=2, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[delta_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=2, col=1)

    fig.add_trace(go.Scatter(x=N_s_2, y=alpha_sim_s, mode='lines', name='simulation'), row=2, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=[alpha_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=2, col=2)

    fig.add_trace(go.Scatter(x=N_s_2, y=mu_sim_s, mode='lines', name='simulation'), row=3, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[mu_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=3, col=1)

    fig.add_trace(go.Scatter(x=N_s_2, y=beta_sim_s, mode='lines', name='simulation'), row=3, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=[beta_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=3, col=2)


    # Graphique de convergence de b
    fig.add_trace(go.Scatter(x=N_s_2, y=b_sim_s, mode='lines', name='simulation'), row=4, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[b_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=4, col=1)

    # Graphique de l'évolution de LL
    fig.add_trace(go.Scatter(x=N_s_2, y=LL_sim_s, mode='lines'), row=4, col=2)

    # Mise en forme du titre et des axes
    fig.update_layout(
        title_text="Convergence et évolution",
        #xaxis_title_text="N_s_2",
        yaxis_title_text="Valeurs",
        showlegend=False,
        width=1000,  # Ajustez la largeur ici
        height=700
    )

    # Centrage du dernier graphique sur la ligne
    #fig.update_yaxes(domain=[0.0, 0.5], row=3, col=1)
    fig.update_xaxes(type="log")
    fig.show()
    return
import plotly.graph_objects as go
def conv_2 (N, nb_inter):
    N_s_2 = np.geomspace(100, N, num=nb_inter, dtype=int)
    kbar = 3
    # Fitting du modèle
    niter = 5
    temperature = 0.5
    stepsize = 0.5
    b_sim_s= []
    m_0_sim_s= []
    gamma_kbar_sim_s= []
    delta_sim_s= []
    beta_sim_s = []
    mu_sim_s = []
    alpha_sim_s = []
    LL_sim_s = []
    #b = 0.60918
    b_sim = 7.7
    m_0_sim = 1.67 
    gamma_kbar_sim  = 0.002
    delta_sim = 0.12
    alpha_sim = 14.2
    mu_sim = 0.002
    beta_sim = -3.2
    returns = simulatedatanig(b_sim,m_0_sim,gamma_kbar_sim,delta_sim, beta_sim, mu_sim, alpha_sim, kbar, N_s_2[-1])
    for i in tqdm(range (len(N_s_2))):
        returns_1 = returns[:N_s_2[i]]
        # Fitting du modèle
        niter = 5
        temperature = 0.5
        stepsize = 0.5
        parameters, LL, niter, output = glo_min_3(kbar, returns_1, niter, temperature, stepsize)
        b_sim_s.append(parameters[0])
        m_0_sim_s.append(parameters[1])
        gamma_kbar_sim_s.append(parameters[2])
        delta_sim_s.append(parameters[3])
        alpha_sim_s.append(parameters[4])
        mu_sim_s.append(parameters[5])
        beta_sim_s.append(parameters[6])
        LL_sim_s.append(LL)
    # Création des subplots avec deux graphiques par ligne
    fig = make_subplots(rows=4, cols=2, subplot_titles=('Convergence de m_0', 'Convergence de gamma_k', 'Convergence de delta', 'Convergence de alpha', 'Convergence de mu', 'Convergence de beta', 'Convergence de b', 'Evolution de LL'),
                        row_heights=[2.5, 2.5,2.5, 2.5])

    # Graphique de convergence de m_0
    fig.add_trace(go.Scatter(x=N_s_2, y=m_0_sim_s, mode='lines', name='simulation'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[m_0_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=1, col=1)

    # Graphique de convergence de gamma_k
    fig.add_trace(go.Scatter(x=N_s_2, y=gamma_kbar_sim_s, mode='lines', name='simulation'), row=1, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=[gamma_kbar_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=1, col=2)

    # Graphique de convergence de delta
    fig.add_trace(go.Scatter(x=N_s_2, y=delta_sim_s, mode='lines', name='simulation'), row=2, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[delta_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=2, col=1)

    fig.add_trace(go.Scatter(x=N_s_2, y=alpha_sim_s, mode='lines', name='simulation'), row=2, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=[alpha_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=2, col=2)

    fig.add_trace(go.Scatter(x=N_s_2, y=mu_sim_s, mode='lines', name='simulation'), row=3, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[mu_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=3, col=1)

    fig.add_trace(go.Scatter(x=N_s_2, y=beta_sim_s, mode='lines', name='simulation'), row=3, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=[beta_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=3, col=2)


    # Graphique de convergence de b
    fig.add_trace(go.Scatter(x=N_s_2, y=b_sim_s, mode='lines', name='simulation'), row=4, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[b_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=4, col=1)

    # Graphique de l'évolution de LL
    fig.add_trace(go.Scatter(x=N_s_2, y=LL_sim_s, mode='lines'), row=4, col=2)

    # Mise en forme du titre et des axes
    fig.update_layout(
        title_text="Convergence et évolution",
        #xaxis_title_text="N_s_2",
        yaxis_title_text="Valeurs",
        showlegend=False,
        width=1000,  # Ajustez la largeur ici
        height=700
    )

    # Centrage du dernier graphique sur la ligne
    #fig.update_yaxes(domain=[0.0, 0.5], row=3, col=1)
    fig.update_xaxes(type="log")
    fig.show()
    return
