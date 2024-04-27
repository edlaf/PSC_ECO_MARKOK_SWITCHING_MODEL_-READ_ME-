import numpy as np

# need to change 

def constructX(x, type_of_polynomial, d):

    if isinstance(x, np.ndarray):
        len1 = len(x)
    else:
        len1 = 1

    if type_of_polynomial.lower() == 'chebychev_firstkind':
        xX = np.ones((len1, d))
        xX[:, 1] = x
        xX[:, 2] = 2*x**2 - 1
        if d > 3:
            xX[:, 3] = 4*x**3 - 3*x
        if d > 4:
            xX[:, 4] = 8*x**4 - 8*x**2 + 1
        if d > 5:
            xX[:, 5] = 16*x**5 - 20*x**3 + 5*x
        if d > 6:
            xX[:, 6] = 32*x**6 - 48*x**4 + 18*x**2 -1
        if d > 7:
            xX[:, 7] = 64*x**7 - 112*x**5 + 56*x**3 - 7*x
    elif type_of_polynomial.lower() == 'chebychev_secondkind':
        xX = np.ones((len1, d))
        xX[:, 1] = 2*x
        xX[:, 2] = 4*x**2-1
        if d > 3:
            xX[:, 3] = 8*x**3 - 4*x
        if d > 4:
            xX[:, 4] = 16*x**4 - 12*x**2 + 1
        if d > 5:
            xX[:, 5] = 32*x**5 - 32*x**3 + 6*x
        if d > 6:
            xX[:, 6] = 64*x**6 - 80*x**4 + 24*x**2 -1
        if d > 7:
            xX[:, 7] = 128*x**7 - 192*x**5 + 80*x**3 - 8*x
        if d > 8:
            xX[:, 8] = 256*x**8 - 448*x**6 + 240*x**4 - 40*x**2 + 1
        if d > 9:
            xX[:, 9] = 512*x**9 - 1024*x**7 + 672*x**5 - 160*x**3 + 10*x
    elif type_of_polynomial.lower() == 'laguerre':
        xX = np.ones((len1, d))
        xX[:, 1] = -x + 1
        xX[:, 2] = ( x**2 - 4*x + 2)/2
        if d > 3:
            xX[:, 3] = (-x**3 + 9*x**2 - 18*x + 6)/6
        if d > 4:
            xX[:, 4] = ( x**4 - 16*x**3 + 72*x**2  - 96*x + 24)/24
        if d > 5:
            xX[:, 5] = (-x**5 + 25*x**4 - 200*x**3 + 600*x**2 - 600*x + 120)/120
        if d > 6:
            xX[:, 6] = ( x**6 - 36*x**5 + 450*x**4 - 2400*x**3 + 5400*x**2 - 4320*x + 720)/720
        if d > 7:
            xX[:, 7] = (-x**7 + 49*x**6 - 882*x**5 + 7350*x**4 - 29400*x**3 + 52920*x**2 - 35280*x + 5040)/5040
    if len1 == 1:
        xX = xX[0]
    return xX