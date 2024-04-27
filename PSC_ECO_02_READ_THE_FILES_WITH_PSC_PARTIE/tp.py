
#fonction tri d une liste par inversion
from numpy import random
liste= random.randint(0,25,30)
print(liste)
def echange(a,b):
    return (b,a)

def tri_insertion(liste):
    for i in range(0, len(liste)-1):
        for i in range(0, len(liste)-1):
            x=liste [i]
            if liste[i+1]>=x:
                liste[i+1],liste[i]=echange(liste[i+1],liste[i])
            i=i+1
            x=liste [i]
    return (liste)
    
print(tri_insertion(liste))

#fonction qui change de base
def base_10A_a (n,a):
    b=[]
    c=0
    for i in range (1,n-1):
        n % a=c
        b=b.append(c)
        n=n//a 
return(b)

base_10A_a(2,2)

#fonction tri en 2 sec qui enleve les doublons
from numpy import random
liste= random.randint(0,25,30)

liste=list(set(liste))
print(liste)

#fonction pythagor
def pythagor(a,b,c):
    L=[]
    for k in range (1,max(a,b,c)):
        for j in range (1,k):
            for i in range (1,j):
                if j**2+i**2 == k**2:
                    L.append(k)
                    L.append(j)
                    L.append(i)
    return (L)

L=pythagor(20,10,10)
print(L)

def pythagor_generalise()


