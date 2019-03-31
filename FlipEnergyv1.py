import numpy as np, numpy.random as rand
from numba import jit

@jit
def FlipEnergy(L,N):
    
    
    site = rand.randint(0,N,size=(2))      #boundary checker/hanlder
    swap = np.argwhere(site==N-1)
    
    if np.any(site==N-1) == True:
        site[swap] = -1
    
    E1 = -1*(L[site[0]+1,site[1]] \
           + L[site[0]-1,site[1]] \
           + L[site[0],site[1]+1] \
           + L[site[0],site[1]-1])*L[site[0],site[1]] #for a single site, can factor out original
    
    E2 =    ( L[site[0]+1,site[1]] \
           +  L[site[0]-1,site[1]] \
           +  L[site[0],site[1]+1] \
           +  L[site[0],site[1]-1])*L[site[0],site[1]]
    
    dE = E2-E1
    
    return dE,site

