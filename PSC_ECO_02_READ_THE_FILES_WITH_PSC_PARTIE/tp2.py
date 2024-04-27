
def base_10A_a (n,a):#fonction qui change de base
    b=[]
    c=0
    for i in range (1,n-1):
        n % a=c
        b=b.append(c)
        n=n//a 
return(b)

base_10A_a(2,2)

