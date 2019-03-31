import numpy as np, numpy.random as rand, matplotlib.pyplot as plt, FlipEnergy as FE, time
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
    
NRuns      = 200
N          = 100 #lattice size
multiplier = 100    
NSample    = 16*multiplier
step       = 0.1*(16/NSample)*N**3       #system scaled step
tTot       = np.int(step*(NSample-1))
tSample    = np.arange(0,tTot+step,step)
T          = 1e-5  # 2.26918531421 = Tc
kB         = 1.38064852e-23
boltz      = 1/(T*kB)
J          = kB
choice     = [-1,1]
k2         = FE.KGrid(N) #offset distance grid in k-space to match the origin of the shifted FFT

lattice     = np.zeros((N,N),dtype=np.int)        #allocation w/ zeros
Avk         = np.zeros((NRuns,NSample),dtype=np.float)
Avkbar      = np.zeros(NSample,dtype=np.float)
L           = np.zeros(NSample,dtype=np.float)

t0= time.time()

for m in range(NRuns):
    print('Working: Run ',m+1,'/',NRuns,'-',100*(m+1)/NRuns,'%')
    lattice = rand.choice(choice,size=(N,N))      #random config setup
    Avk[m,0] = FE.LatticeFFT(lattice,k2)
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
        Avk[m,y+1] = FE.LatticeFFT(lattice,k2)
        
Avkbar = np.mean(Avk,axis=0,dtype=np.float64)
Avkmax = np.amax(Avk,axis=0)
Avkmin = np.amin(Avk,axis=0)

Lmax = np.log(1/Avkmax[1:])
Lmin = np.log(1/Avkmin[1:])


L = 1/Avkbar
logL=np.log(L[1:])
logt=np.log(tSample[1:])

fig1 = plt.figure(1)

Fpopt,Fpcov = fit(funcfit,logt,logL)
Bpopt,Bpcov = fit(funcfit,logt,logL,bounds=([-100,0.5],[100,0.501]))
B2popt,B2pcov = fit(funcfit,logt,logL,bounds=([-100,1/3],[100,1000001/3000000]))

plt.plot(logt,funcfit(logt,*Fpopt),'c-',label='Average gradient of data',linewidth=3)
plt.fill_between(logt,Lmin,Lmax)
plt.plot(logt,funcfit(logt,*Bpopt),'g-',label='Scaling Hypothesis Relation',linewidth=3)
plt.plot(logt,logL,'r-',label='Simulated data')
plt.plot(logt,funcfit(logt,*B2popt),'k',label='Other MC Results',linewidth=3)
plt.xlabel('Ln(t)')
plt.ylabel('Ln(L)')
#plt.legend(loc='best')

t1    = time.time()
total = t1-t0

print('Time taken: ',total,' for ',NRuns,' runs.')
print('Time per run: ',total/NRuns)
print('Total choices: ',tTot*NRuns)
print('Average time taken per choice: ',total/tTot)
