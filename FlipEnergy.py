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

@jit
def KGrid(N):
    k2 = np.zeros((N,N),dtype=np.float)
    kx=ky= np.arange(0,(N+2)/2,1,dtype=int)
    kxg,kyg= np.meshgrid(kx,ky)
    kg= (kxg**2+kyg**2)**0.5
    k2[(N//2):N,(N//2):N] = kg[0:len(kx)-1,0:len(kx)-1]
    k2[0:(N//2)+1,(N//2):N] = np.rot90(kg[0:len(kx)-1,0:len(kx)])
    k2[0:(N//2)+1,0:(N//2)+1] = np.rot90(kg,2)
    k2[(N//2):N,0:(N//2)+1] = np.rot90(kg[:,0:len(kx)-1],3)
    
    return k2
    
@jit    
def LatticeFFT(lat,grid):
    
    fft=np.fft.fft2(lat)
    fft*=np.conj(fft)
    fft2=np.fft.fftshift(fft)
    avk=np.sum(grid*fft2)/np.sum(fft2)
    return avk
    
    