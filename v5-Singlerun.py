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
#    -v5 Single run; 
#        -removed boltzmann factor probability check for speed, (doesn't actually do anything)
#    -v5 Multiple run;
#        -Moved fourier transform code into separate function in FlipEnergy (will rename soon)
#        
#TODO:
#    -'v5 implementation' ie. separate codes for single and multiple runs
#        -single run will have animation, full plots for system progression etc. 
#        -multiple runs will average all Ls, for the same T, and be fit to curve.
#            -v5 single:
#                -save animations, and system progression plots
#                

def funcfit(x,a,b):
    return b*x+a
    
def init():
    map1.set_array(latticesave[:,:,0])
    return map1,
    
def animate(k):
    map1.set_array(latticesave[:,:,k])
    return map1, 

N          = int(400) #lattice size
multiplier = 1      #affects number of frames in animation, plus data points for fig 3 plot.
NSample    = 16*multiplier
step       = 10*(16/NSample)*N**2       #system scaled step
tTot       = np.int(step*(NSample-1))
tSample    = np.arange(0,tTot+step,step)

kB         = 1.38064852e-23
T          = 1e-5  # 2.26918531421 = Tc*kb/J
boltz      = 1/(T*kB)
J          = kB
choice     = [-1,1]
k2         = FE.KGrid(N) #offset distance grid in k-space to match the origin of the shifted FFT

lattice     = np.zeros((N,N),dtype=np.int)        #allocation w/ zeros
latticesave = np.zeros((N,N,NSample),dtype=np.int)

Avk         = np.zeros(NSample,dtype=np.float)
L           = np.zeros(NSample,dtype=np.float)


lattice = rand.choice(choice,size=(N,N))      #random config setup
latticesave[:,:,0] = lattice                  #first lattice of each system is the starting lattice

t0= time.time()

for y in range(int(NSample-1)):    
    for t in range(int(step)):  
        dE,spinsite = FE.FlipEnergy(lattice,N)        
        if (dE > 0):
                pcheck = rand.ranf()
                p=np.exp(-dE/T)
                if pcheck < p:
                    lattice[spinsite[0],spinsite[1]] *= -1
        else:
            lattice[spinsite[0],spinsite[1]] *= -1
    
    latticesave[:,:,y+1] = lattice
    print(100*(y+2)/NSample,'%')

fig1 = plt.figure(1)
ax1  = fig1.add_axes([0,0,1,1],aspect='equal',xlim=(0,N),ylim=(0,N))
cmap = plt.get_cmap('Greys')
map1 = ax1.imshow(latticesave[:,:,0],cmap=cmap,interpolation='none',animated=True)
ax1.axis('off')
ani = anim.FuncAnimation(fig1,animate,init_func=init,frames=NSample,interval=(350/multiplier))#,blit=True)

plt.show()


        
fig2, ax = plt.subplots(4,4)
fig2.subplots_adjust(top=1,bottom=0,left=0.1,right=0.9,wspace=0,hspace=0.05)
ax = ax.ravel()

for i in range(np.int(NSample/multiplier)):
    ax[i].imshow(latticesave[:,:,i*multiplier],cmap=cmap,interpolation='none')
    ax[i].axis('off')
    
for i in range(np.int(NSample)):
    Avk[i]   =  FE.LatticeFFT(latticesave[:,:,i],k2)
    
L = 1/Avk
logL=np.log(L[1:])
logt=np.log(tSample[1:]/step)

fig3=plt.figure(3)
plt.plot(logt,logL,'r-',label='data')
popt,pcov = fit(funcfit,logt,logL,bounds=([-1000,0.49],[1000,0.51]))
plt.plot(logt,funcfit(logt,*popt),label='fit')
plt.xlabel('Ln(t)')
plt.ylabel('Ln(L)')
plt.legend(loc='best')

t1    = time.time()
total = t1-t0

print('Time taken: ',total)
print('Total choices: ',tTot)
print('Average time taken per choice: ',total/tTot)

#name = input('Input video name: ')
#FFwriter = anim.FFMpegWriter(fps = 30,extra_args=['-vcodec','libx264','-profile','high444'])    
#ani.save(name+'.mp4',writer=FFwriter) 