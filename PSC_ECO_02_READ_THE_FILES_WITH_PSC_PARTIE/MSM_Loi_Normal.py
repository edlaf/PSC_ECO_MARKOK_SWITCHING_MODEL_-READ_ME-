import numpy as np
import pandas as pd
import scipy.optimize as opt
from statsmodels import regression
import statsmodels.formula.api as sm
from numba import jit, njit, prange, float64, int64
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as sc
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

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
    # Presents as: [b, m0, gamma_kbar, sigma]
    x0 = theta

    # basinhopping arguments
    niter = niter
    T = temperature
    stepsize = stepsize
    args = (kbar, data)

    # bounds
    bounds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-4,5))

    # minimizer_kwargs
    minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds, args = args)

    res = opt.basinhopping(
                func = f, x0 = x0, niter = niter, T = T, stepsize = stepsize,
                minimizer_kwargs = minimizer_kwargs)

    parameters, LL, niter, output = res.x,res.fun,res.nit,res.message

    return(parameters, LL, niter, output)


def loc_min(kbar, data):
    """step 1: local minimization
       parameter estimation uses bounded optimization (scipy.optimize.fminbound)
    """

    # set up
    b = np.array([1.5, 5, 15, 30])
    lb = len(b)
    gamma_kbar = np.array([0.1, 0.5, 0.9, 0.95])
    lg = len(gamma_kbar)
    sigma = np.std(data)

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
            theta_in = [b[i], gamma_kbar[j], sigma]
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

    theta = theta_out[idx[0],:].tolist()+[sigma]
    theta_out = theta_out[idx,:]

    return(theta, theta_LLs, theta_out, ierr, numfunc)


def g_LL(m0, kbar, data, theta_in):
    """return LL, the vector of log likelihoods
    """

    # set up
    b = theta_in[0]
    gamma_kbar = theta_in[1]
    sigma = theta_in[2]
    kbar2 = 2**kbar
    T = len(data)
    pa = (2*np.pi)**(-0.5)

    # gammas and transition probabilities
    A = g_t(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p(kbar, m0)

    # volatility model
    s = sigma*g_m

    # returns
    w_t = data
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16

    # log likelihood using numba
    LL = _LL(kbar2, T, A, g_m, w_t)

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

    return(LL)


def g_pi_t(m0, kbar, data, theta_in):
    """return pi_t, the current distribution of states
    """

    # set up
    b = theta_in[0]
    gamma_kbar = theta_in[1]
    sigma = theta_in[2]
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
    s = sigma*g_m

    # returns
    w_t = data
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
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
    sigma = theta[3]

    theta_in = [b, gamma_kbar, sigma]

    return(theta_in)

# simulation de la data du processus MSM
def simulatedata(b,m0,gamma_kbar,sig,kbar,T):
 
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
    dat = np.sqrt(dat)*sig* np.random.normal(size = T)   # VOL TIME SCALING
    dat = dat.reshape(-1,1)
 
    return(dat)

def rep(name):
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    # Télécharger les données historiques pour une action spécifique
    ticker_symbol = name  # Symbole boursier
    start_date = "1900-07-05"
    end_date = "2024-01-01"

    # Récupérer les données historiques
    data = yf.download(ticker_symbol, start=start_date, end=end_date)#, interval ='60m')
    

    # Calculer les rendements
    data['Returns'] = data['Adj Close'].pct_change()
    returns_numpy = data['Returns'].to_numpy()[1:]

    # Calculer les rendements
    data['Returns'] = data['Adj Close'].pct_change()
    print(data)
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    returns_numpy = returns_numpy.reshape(-1,1)
    T = 2000
    kbar = 5
    niter = 5
    temperature = 0.5
    stepsize = 0.5

    parameters, LL, niter, output = glo_min(kbar, returns_numpy, niter, temperature, stepsize)

    # parameters contient les valeurs estimées
    # name parameters for later use:
    # name parameters for later use:
    b_sim = parameters[0]
    m_0_sim = parameters[1]
    gamma_kbar_sim = parameters[2]
    sigma_sim = parameters[3]

    LL_sim = LL



    #print("Valeurs mises au depart :", "b =", b,"m_0=", m0, "gamma_kbar =",gamma_kbar,"delta =", delta, "alpha =",alpha, "beta =", beta, "mu =", mu)
    print("Parameters from glo_min for Simulated dataset: ", "\n"
        "kbar = ", kbar,"\n"
        'b = %.5f' % b_sim,"\n"
        'm_0 = %.5f' % m_0_sim,"\n"
        'gamma_kbar = %.5f' % gamma_kbar_sim,"\n"
        'sigma = %.5f' % (sigma_sim),"\n"
        'Likelihood = %.5f' % LL_sim,"\n"
        "niter = " , niter,"\n"
        "output = " , output,"\n")
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    T = len(returns_numpy)+1
    
    #data simulés
    data_sim = simulatedata(b_sim,m_0_sim,gamma_kbar_sim,sigma_sim, kbar, T).reshape(-1)
    fig = go.Figure()
    # Représentaion des données
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

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data['Returns'], nbinsx=200, xbins=dict(size=0.002), histnorm='probability', name='Returns réels', opacity=0.7))
    fig.add_trace(go.Histogram(x=data_sim, nbinsx=200, xbins=dict(size=0.002), histnorm='probability', name='Returns simulés', opacity=0.7))

    fig.update_layout(
        title='Historigramme comparatif des returns réels et simulés (' + ticker_symbol + ')',
        xaxis_title='Valeur des returns',
        yaxis_title='Répartition des returns',
        legend=dict(x=0.01, y=0.99),
        height=600,
        width=1000,
        bargap=0.0,
        font=dict(size=20)
    )

    fig.show()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    ret = returns_numpy.reshape(-1,1).reshape(-1)
    fig = sm.qqplot(np.sort(ret), line='s', color='green', scale=2)
    plt.grid(True)
    plt.title('Returns bruts')
    plt.show()
    sm.qqplot(np.sort(data_sim), line='s', color = 'red', scale=2)
    plt.grid(True)
    plt.title('Log-returns')
    plt.show()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")

from tqdm import tqdm

def analyse_data(tab):
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    L = len(tab)
    kbar_ = [1,2,3, 4, 5, 6]
    analyse = [[],[],[],[],[]]# b_sim, M_0, gamma, delta, alpha, mu, beta, LL
    index   = ['b', 'm_0', 'gamma', 'sigma','LL']
    columns = ['kbar = 1','kbar = 2','kbar = 3', 'kbar = 4', 'kbar = 5', 'kbar = 6']
    for i in range (len(tab)):
        # Télécharger les données historiques pour une action spécifique
        ticker_symbol = tab[i]  # Symbole boursier d'Apple Inc. (AAPL)
        start_date = "1900-01-01"
        end_date = "2024-01-01"

        # Récupérer les données historiques
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        # Calculer les rendements
        data['Returns'] = data['Adj Close'].pct_change()
        # Extraire les rendements sous forme de tableau NumPy
        returns_numpy = data['Returns'].to_numpy()[1:]
        returns_numpy = returns_numpy.reshape(-1,1)
        for j in tqdm(range (len(kbar_))):
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
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    return df.transpose()

def residus_1(name, horizon, kbar, nb_inter):
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    Forcast = []
    ticker_symbol      = name  # Symbole boursier
    start_date = "2003-01-01"
    end_date = "2024-01-01"
    # Récupérer les données historiques
    data               = yf.download(ticker_symbol, start=start_date, end=end_date)
    data['Returns']    = data['Adj Close'].pct_change()
    returns_numpy      = data['Returns'].to_numpy()[1:]
    returns_numpy      = returns_numpy.reshape(-1,1)
    #print(len(returns_numpy))
    for i in range (horizon):
        returns_numpy  = returns_numpy[:len(returns_numpy)-horizon+i]
        # Fitting du modèle
        niter          = 8
        temperature    = 0.2
        stepsize       = 0.2
        parameters, LL, niter, output = glo_min(kbar, returns_numpy, niter, temperature, stepsize)
        print("Step ",i+1, " :", "-----------------------------------fitting done-----------------------------------" )

        # parameters contient les valeurs estimées
        # name parameters for later use:
        b_sim = parameters[0]
        m_0_sim = parameters[1]
        gamma_kbar_sim = parameters[2]
        sigma_sim = parameters[3]
        LL_sim = LL
        '''
        print("Parameters from glo_min for Simulated dataset: ", "\n"
        "kbar = ", kbar,"\n"
        'b = %.5f' % b_sim,"\n"
        'm_0 = %.5f' % m_0_sim,"\n"
        'gamma_kbar = %.5f' % gamma_kbar_sim,"\n"
        'sigma = %.5f' % (sigma_sim),"\n")
        print("Value at minimum :",LL_sim)
        '''
        returns        = simulatedata(b_sim,m_0_sim,gamma_kbar_sim,sigma_sim, kbar, 1).reshape(-1)[0]
        for _ in range (nb_inter):
            returns+=simulatedata(b_sim,m_0_sim,gamma_kbar_sim,sigma_sim, kbar, 1).reshape(-1)[0]
        returns = returns/nb_inter
        prix           = data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon+i:][0]
        #print("prix :", prix)
        p              = np.exp(returns)
        Forcast.append(prix*p)
    #print("reel :",data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:])
    MAE  = 0
    MSE  = 0
    R    = 0
    le = len(Forcast)
    moy = np.mean(data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:])
    for j in range (len(Forcast)):
        MAE += np.abs(Forcast[j]-data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:][j])/le
        MSE += (Forcast[j]-data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:][j])**2/le
        R +=(Forcast[j]-moy)**2
    R = 1 - MSE*le/R
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    print("Forecast :",Forcast)
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    
    T = np.arange(0,horizon,1)
    plt.figure(figsize=(20,12))
    plt.plot(T, Forcast, label = "forcast")
    plt.plot(T, data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:], label = "reel")
    plt.title("Simulations par fitting journalier")
    plt.legend()
    plt.show()
    print("Résidu MAE :", MAE)
    print("Résidu MSE :", MSE)
    print("Résidu R**2 :", R)
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    return Forcast, data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:], MAE, MSE, R

def residus_2(name, horizon, kbar, nb_inter):
    Forcast = []
    ticker_symbol      = name  # Symbole boursier
    start_date = "2003-01-01"
    end_date = "2024-01-01"
    # Récupérer les données historiques
    data               = yf.download(ticker_symbol, start=start_date, end=end_date)
    data['Returns']    = data['Adj Close'].pct_change()
    returns_numpy      = data['Returns'].to_numpy()[1:]
    returns_numpy      = returns_numpy.reshape(-1,1)
    #print(len(returns_numpy))
    for i in range (horizon):
        returns_numpy  = returns_numpy[:len(returns_numpy)-horizon+i]
        # Fitting du modèle
        niter          = 8
        temperature    = 0.2
        stepsize       = 0.2
        parameters, LL, niter, output = glo_min(kbar, returns_numpy, niter, temperature, stepsize)
        print("Step ",i+1, " :", "-----------------------------------fitting done-----------------------------------" )

        # parameters contient les valeurs estimées
        # name parameters for later use:
        b_sim = parameters[0]
        m_0_sim = parameters[1]
        gamma_kbar_sim = parameters[2]
        sigma_sim = parameters[3]
        LL_sim = LL
        '''
        print("Parameters from glo_min for Simulated dataset: ", "\n"
        "kbar = ", kbar,"\n"
        'b = %.5f' % b_sim,"\n"
        'm_0 = %.5f' % m_0_sim,"\n"
        'gamma_kbar = %.5f' % gamma_kbar_sim,"\n"
        'sigma = %.5f' % (sigma_sim),"\n")
        print("Value at minimum :",LL_sim)
        '''
        returns        = simulatedata(b_sim,m_0_sim,gamma_kbar_sim,sigma_sim, kbar, 1).reshape(-1)[0]
        for _ in range (nb_inter):
            returns+=simulatedata(b_sim,m_0_sim,gamma_kbar_sim,sigma_sim, kbar, 1).reshape(-1)[0]
        returns = returns/(nb_inter+1)
        prix           = data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon+i:][0]
        #print("prix :", prix)
        p              = np.exp(returns)
        Forcast.append(prix*p)
    #print("reel :",data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:])
    MAE  = 0
    MSE  = 0
    R    = 0
    le = len(Forcast)
    moy = np.mean(data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:])
    for j in range (len(Forcast)):
        MAE += np.abs(Forcast[j]-data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:][j])/le
        MSE += (Forcast[j]-data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:][j])**2/le
        R +=(Forcast[j]-moy)**2
    R = 1 - MSE*le/R

    return Forcast, data['Close'].to_numpy()[1:][len(data['Close'].to_numpy()[1:])-horizon:], MAE, MSE, R
def convergence (N, N_2,name, horizon, kbar_max, nb_inter):
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    N_s = np.geomspace(1, N, num=nb_inter, dtype=int)
    ticker_symbol      = name  # Symbole boursier
    start_date = "2022-07-05"
    end_date = "2024-01-01"
    # Récupérer les données historiques
    data               = yf.download(ticker_symbol, start=start_date, end=end_date, interval ='60m')
    data['Returns']    = data['Adj Close'].pct_change()
    returns_numpy      = data['Returns'].to_numpy()[1:]
    returns_numpy      = returns_numpy.reshape(-1,1)
    N_s_2 = np.geomspace(10, N_2, num=nb_inter, dtype=int)
    MAE_s = []
    MSE_s = []
    R_s = []
    for i in tqdm(range (len(N_s))):
        forcast, returns, MAE, MSE, R = residus_2(name, horizon, kbar_max, N_s[i])
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
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    # moments
    moments_1 = []
    moments_2 = []
    moments_3 = []
    # Fitting du modèle
    niter          = 8
    temperature    = 0.2
    stepsize       = 0.2
    parameters, LL, niter, output = glo_min(kbar_max, returns_numpy, niter, temperature, stepsize)
    #print( "-----------------------------------fitting done-----------------------------------" )
        # parameters contient les valeurs estimées
        # name parameters for later use:
    b_sim = parameters[0]
    m_0_sim = parameters[1]
    gamma_kbar_sim = parameters[2]
    sigma_sim = parameters[3]
    LL_sim = LL
    
    for i in tqdm(range (len(N_s_2))):
        returns_1 = simulatedata(b_sim,m_0_sim,gamma_kbar_sim,sigma_sim, kbar_max, N_s_2[i]).reshape(-1)
        moments_1.append(np.mean(returns_1))
        moments_2.append(np.std(returns_1))
        moments_3.append(sc.skew(returns_1))
    #print(moments_1)
    #print(moments_2)
    #print(moments_3)
    M_1 = [np.mean(returns_numpy) for _ in range (len(N_s_2))]
    M_2 = [np.std(returns_numpy) for _ in range (len(N_s_2))]
    M_3 = [sc.skew(returns_numpy)[0] for _ in range (len(N_s_2))]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Convergence des moments d'ordre 1", "Convergence des moments d'ordre 2", "Convergence des moments d'ordre 3"))

    # Add traces
    fig.add_trace(go.Scatter(x=N_s_2, y=moments_1, mode='lines', name='Moyenne obtenue'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=M_1, mode='lines', name='Moyenne réelle'), row=1, col=1)

    fig.add_trace(go.Scatter(x=N_s_2, y=moments_2, mode='lines', name='Variance'), row=1, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=M_2, mode='lines', name='Variance réelle'), row=1, col=2)

    fig.add_trace(go.Scatter(x=N_s_2, y=moments_3, mode='lines', name='Skewness'), row=2, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=M_3, mode='lines', name='Skewness réelle'), row=2, col=1)

    # Update layout
    fig.update_layout(height=600, width=1000, title_text="Convergence des moments d'ordre 1, 2 et 3")
    fig.update_xaxes(type="log")
    # Show plot
    fig.show()
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    return

def conv_2 (N, kbar,nb_inter):
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    N_s_2 = np.geomspace(100, N, num=nb_inter, dtype=int)
    # Paramétrisation
    niter          = 8
    temperature    = 0.2
    stepsize       = 0.2
    b_sim_s= []
    b_sim =5
    m_0_sim_s= []
    m_0_sim = 1.6
    gamma_kbar_sim_s= []
    gamma_kbar_sim = 0.7
    sigma_sim_s= []
    sigma_sim = 0.02
    LL_sim_s = []
    returns = simulatedata(b_sim,m_0_sim,gamma_kbar_sim,sigma_sim, kbar, N_s_2[-1])
    for i in tqdm(range (len(N_s_2))):
        returns_1 = returns[:N_s_2[i]]
        # Fitting du modèle
        niter          = 8
        temperature    = 0.2
        stepsize       = 0.2
        parameters, LL, niter, output = glo_min(kbar, returns_1, niter, temperature, stepsize)
        b_sim_s.append(parameters[0])
        m_0_sim_s.append(parameters[1])
        gamma_kbar_sim_s.append(parameters[2])
        sigma_sim_s.append(parameters[3])
        LL_sim_s.append(LL)

    # Création des subplots avec deux graphiques par ligne
    fig = make_subplots(rows=3, cols=2, subplot_titles=('Convergence de m_0', 'Convergence de gamma_k', 'Convergence de sigma', 'Convergence de b', 'Evolution de LL'),
                        row_heights=[2.5, 2.5, 2.5])

    # Graphique de convergence de m_0
    fig.add_trace(go.Scatter(x=N_s_2, y=m_0_sim_s, mode='lines', name='simulation'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[m_0_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=1, col=1)

    # Graphique de convergence de gamma_k
    fig.add_trace(go.Scatter(x=N_s_2, y=gamma_kbar_sim_s, mode='lines', name='simulation'), row=1, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=[gamma_kbar_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=1, col=2)

    # Graphique de convergence de sigma
    fig.add_trace(go.Scatter(x=N_s_2, y=sigma_sim_s, mode='lines', name='simulation'), row=2, col=1)
    fig.add_trace(go.Scatter(x=N_s_2, y=[sigma_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=2, col=1)

    # Graphique de convergence de b
    fig.add_trace(go.Scatter(x=N_s_2, y=b_sim_s, mode='lines', name='simulation'), row=2, col=2)
    fig.add_trace(go.Scatter(x=N_s_2, y=[b_sim for _ in range(len(N_s_2))], mode='lines', name='valeur théorique'), row=2, col=2)


    # Graphique de l'évolution de LL
    fig.add_trace(go.Scatter(x=N_s_2, y=LL_sim_s, mode='lines'), row=3, col=1)

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
    print("___________________________________________________________________________________________________________________")
    print("___________________________________________________________________________________________________________________")
    return
    
