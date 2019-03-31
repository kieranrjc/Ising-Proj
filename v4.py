import numpy as np, numpy.random as rand, matplotlib.pyplot as plt, FlipEnergy as FE, time, numpy.fft as fft
from matplotlib import animation as anim  
from scipy.optimize import curve_fit as fit
plt.close('all')


#changelog:
#    -v1 base code outlined, flip energy measured, animated for each choice
#    -v2 energy flips moved to separate module, (FlipEnergy.py)
#    -v3 animation added, code sped up by not updating every frame choice, 
#    -v4 implemented data handling for length scale equations, k-grid, sums and plot;
#        also implemented multiplier system to allow for continuity at different amounts of samples for data
#            - all multiples of 16 for the 4x4 'system progression' plot
#            - step sized reduced, so always the same number of steps for 16, at a multiplier of 1
#                -TODO implement N-scaling for step, (larger N systems need more time to equilibrate)
#            - animation speed is scaled based on multiplier for more consistent animations. 
#        
#TODO:
#    -'v5 implementation' ie. separate codes for single and multiple runs
#        -single run will have animation, full plots for system progression etc. 
#        -multiple runs will average all Ls, for the same T, and be fit to curve.
#            -v5 single:
#                -save animations, and system progression plots
#                -export some verbose functions, such as the making of k2, export to kgrid can put under flip energy
#            -v5 multiple:
#                -remove animation and plots bar fig3 ln(L)/ln(t) plot. (possibly move them to own function in single)
#                -set Nruns, (allocation inside loop, stationary params outside loop)
#                -averaging and fitting for the loop.
        
    
            
    


def funcfit(x,a,b):
    return b*x+a

def init():
    map1.set_array(latticesave[:,:,0])
    return map1,
    
def animate(k):
    map1.set_array(latticesave[:,:,k])
    return map1, 

N          = 50 #lattice size
multiplier = 10
NSample    = 16*multiplier
step       = (16/NSample)*N**(2+N/1e3)       #system scaled step
tTot       = np.int(step*(NSample-1))
tSample    = np.arange(0,tTot+step,step)
T          = 2.26918531421  # 2.26918531421 = Tc
kB         = 1.38064852e-23
boltz      = 1/(T*kB)
J          = 1
choice     = [-1,1]
k2         = FE.KGrid(N) #offset distance grid in k-space to match the origin of the shifted FFT

lattice     = np.zeros((N,N),dtype=np.int)        #allocation w/ zeros
latticesave = np.zeros((N,N,NSample),dtype=np.int)
latticesfft = np.empty((NSample, 0)).tolist()
Avk         = np.zeros(NSample,dtype=np.float)
L           = np.zeros(NSample,dtype=np.float)


lattice = rand.choice(choice,size=(N,N))      #random config setup
latticesave[:,:,0] = lattice                  #first lattice of each system is the starting lattice

t0= time.time()

for y in range(int(NSample-1)):    
    for t in range(int(step)):  
        dE,spinsite = FE.FlipEnergy(lattice,N)        
        dE *= J
        
        p = np.exp(-dE*boltz)
        pcheck = rand.ranf()
        
        if (dE < 0) or (pcheck < p) :
            lattice[spinsite[0],spinsite[1]] *= -1
    
    latticesave[:,:,y+1] = lattice

fig1 = plt.figure(1)
ax1  = fig1.add_axes([0,0,1,1],aspect='equal',xlim=(0,N),ylim=(0,N))
cmap = plt.get_cmap('Greys')
map1 = ax1.imshow(latticesave[:,:,0],cmap=cmap,interpolation='none',animated=True)
anim = anim.FuncAnimation(fig1,animate,init_func=init,frames=NSample,interval=(350/multiplier),blit=True)
ax1.axis('off')
plt.show()
        
fig2, ax = plt.subplots(4,4)
fig2.subplots_adjust(top=1,bottom=0,left=0.1,right=0.9,wspace=0,hspace=0.05)
ax = ax.ravel()

for i in range(np.int(NSample/multiplier)):
    ax[i].imshow(latticesave[:,:,i*multiplier],cmap=cmap,interpolation='none')
    ax[i].axis('off')
    
for i in range(np.int(NSample)):
    latticesfft[i]  = fft.fft2(latticesave[:,:,i])
    latticesfft[i] *= np.conj(latticesfft[i])
    latticesfft[i]  = fft.fftshift(latticesfft[i])
    Avk[i]          = np.sum(k2*latticesfft[i])/np.sum(latticesfft[i])    
    
L = 1/Avk
logL=np.log(L[1:])
logt=np.log(tSample[1:])

fig3=plt.figure(3)
plt.plot(logt,logL,'rx',label='data')
popt,pcov = fit(funcfit,logt,logL,bounds=([-10,0.49],[0,0.51]))
plt.plot(logt,funcfit(logt,*popt),label='fit')
plt.xlabel('Ln(t)')
plt.ylabel('Ln(L)')
plt.legend(loc='best')

    


t1    = time.time()
total = t1-t0

print('Time taken: ',total)
print('Total choices: ',tTot)
print('Average time taken per choice: ',total/tTot)
