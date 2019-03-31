import numpy as np, numpy.random as rand, matplotlib.pyplot as plt, FlipEnergyv1 as FE
from numba import jit 
plt.close('all')



N = 25  #lattice size
T = 2.2
kB = 1 # 1.38064852e-23
boltz = 1/(T*kB)
z = 4
J = 1

choice = [-1,1]
lattice = np.zeros((N,N),dtype=np.int)        #allocation
lattice = rand.choice(choice,size=(N,N))      #random config setup

fig1 = plt.figure(1)
ax1 = fig1.add_axes([0,0,1,1],aspect='equal',xlim=(0,N),ylim=(0,N))
cmap = plt.get_cmap('Greys')
ax1.axis('off')
map1 = ax1.imshow(np.rot90(lattice.T),cmap=cmap,interpolation='none')


for t in range(int(1e8)):
    dE,spinsite = FE.FlipEnergy(lattice,N)
    
    dE *= J
    
    p = np.exp(-dE*boltz)
    pcheck = rand.ranf()
    
    if (dE < 0) or (pcheck < p) :
        lattice[spinsite[0],spinsite[1]] *= -1
    
    
    plt.pause(1e-12)
    map1.set_array(np.tile(np.rot90(lattice),(3,3)))        #np.tile to check boundary conditions
    fig1.canvas.draw()