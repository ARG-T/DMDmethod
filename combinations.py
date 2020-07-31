# 結構速い(？)

def combinations(x,y,N):
        fac = [0]*(N+1)
        fac[0],fac[1] = 1,1
        
        for i in range(2,N+1):
            fac[i] = (fac[i-1]*i)%MOD
            
        return (fac[x+y]*pow(fac[x],MOD-2,MOD)*pow(fac[y],MOD-2,MOD))%MOD

# それなりに速い
