import numpy as np, numpy.random as rand, matplotlib.pyplot as plt, time 
from numba import jit 
plt.close('all')



N = 200   #lattice size
T = 2.26918531421  # 2.26918531421 = Tc
kB =  1.38064852e-23
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

t0= time.time()


for t in range(int(1e6)):
    spinsite = rand.randint(0,N,size=(2))
    swap = np.argwhere(spinsite==np.int64(N-1))
    
    if np.any(spinsite==N-1) == True:
        spinsite[swap] = -1
    
    E1 = -J*(lattice[spinsite[0]+1,spinsite[1]] \
        +    lattice[spinsite[0]-1,spinsite[1]] \
        +    lattice[spinsite[0],spinsite[1]+1] \
        +    lattice[spinsite[0],spinsite[1]-1])*lattice[spinsite[0],spinsite[1]]
    
    E2 = J*(   lattice[spinsite[0]+1,spinsite[1]] \
           +    lattice[spinsite[0]-1,spinsite[1]] \
           +    lattice[spinsite[0],spinsite[1]+1] \
           +    lattice[spinsite[0],spinsite[1]-1])*lattice[spinsite[0],spinsite[1]]
    
    dE = E2-E1
    
    p = np.exp(-dE*boltz)
    pcheck = rand.ranf()
    
    if (dE < 0) or (pcheck < p) :
        lattice[spinsite[0],spinsite[1]] *= -1
    

plt.pause(1e-12)
map1.set_array(np.tile(np.rot90(lattice),(3,3))) 
fig1.canvas.draw()
    
t1 = time.time()
total = t1-t0
print(total)
    