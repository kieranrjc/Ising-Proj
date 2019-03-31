import numpy as np, numpy.random as rand, matplotlib.pyplot as plt, FlipEnergy as FE, time, numpy.fft as fft
from matplotlib import animation as anim  
plt.close('all')

def init():
    map1.set_array(latticesave[:,:,0])
    return map1,
    
def animate(k):
    map1.set_array(latticesave[:,:,k])
    return map1, 

N = 100 #lattice size
NSample = 32
T = 2.26918531421  # 2.26918531421 = Tc
kB =  1.38064852e-23
boltz = 1/(T*kB)
J = 1
choice = [-1,1]
lattice = np.zeros((N,N),dtype=np.int)        #allocation
latticesave = np.zeros((N,N,NSample),dtype=np.int)
latticesfft = np.empty((NSample, 0)).tolist()

lattice = rand.choice(choice,size=(N,N))      #random config setup
latticesave[:,:,0] = lattice

t0= time.time()

for y in range(int(NSample-1)):    
    for t in range(int(0.5*(N**2))):  
        dE,spinsite = FE.FlipEnergy(lattice,N)        
        dE *= J
        
        p = np.exp(-dE*boltz)
        pcheck = rand.ranf()
        
        if (dE < 0) or (pcheck < p) :
            lattice[spinsite[0],spinsite[1]] *= -1
    
    latticesave[:,:,y+1] = lattice

fig1 = plt.figure(1)
ax1 = fig1.add_axes([0,0,1,1],aspect='equal',xlim=(0,N),ylim=(0,N))
cmap = plt.get_cmap('Greys')
ax1.axis('off')
map1 = ax1.imshow(latticesave[:,:,0],cmap=cmap,interpolation='none',animated=True)
anim = anim.FuncAnimation(fig1,animate,init_func=init,frames=NSample,interval=50)
plt.show()
        
fig2, ax = plt.subplots(4,4)
fig2.subplots_adjust(top=1,bottom=0,left=0.1,right=0.9,wspace=0,hspace=0.05)
ax = ax.ravel()

for i in range(np.int(NSample/2)):
    ax[i].imshow(latticesave[:,:,i*2],cmap=cmap,interpolation='none')
    ax[i].axis('off')
    
for i in range(np.int(NSample)):
    latticesfft[i]=fft.fft2(latticesave[:,:,i])
    latticesfft[i]*=np.conj(latticesfft[i])
    
latticesfft=np.asarray(latticesfft)


t1 = time.time()
total = t1-t0

print('Time taken: ',total)
print('Total choices: ',np.int(0.5*N**2*(NSample-1)))
print('Average time taken per choice: ',total/(np.int(0.5*N**2*(NSample-1))))
