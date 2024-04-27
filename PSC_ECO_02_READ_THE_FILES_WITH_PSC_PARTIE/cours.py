def base_10A_a (n,a):
    b=[]
    c=0
    while n!=0:
        c=n%a
        b.append(c)
        n=n//a 
    return(b)

a=base_10A_a(100,2)
print(a)


def base_10A_a (n,a):
    b=[]
    c=0
    while n!=0:
        c=n%a
        b.append(c)
        n=n//a 
    #for i in range(0,len(b)-1):
       # b[i]=b([len(b)-i-1]
    return(b)
    
a=base_10A_a(145,2)
print(a)



