## Projet_2 De-Vries equation

import numpy as np
import numpy.fft as fft
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#définition des constantes
L = 50
N = 256
tmax = 60  # faire pour 30, 40, 50, 60, 70 
dt = 0.0004
Nt = int(tmax/dt)
x = np.linspace(0, L, N)
t = np.linspace(0, tmax, Nt)
k = np.linspace(-(N/2), (N/2)-1, N)
j = complex(0,1)

c_1 = 0.75
a_1 = 0.33
c_2 = 0.4
a_2 = 0.65

#création espace des solution
sol = np.zeros((len(x), len(t)))
u = np.zeros((len(x), len(t)))

#condition initiale
u_0 = (c_1/2) * (np.cosh((np.sqrt(c_1)/ 2) * (x - a_1 *L)))**(-2) + (c_2/ 2) * (np.cosh((np.sqrt(c_2)/ 2) * (x - a_2 * L)))**(-2)
u[:,0] = u_0

def Solve(u, sol):
    #définition des different espace des fonctions utilisée
    ustar = np.zeros((len(x), len(t)), dtype=complex)
    g = np.zeros((len(x), len(t)))
    gstar = np.zeros((len(x), len(t)),dtype=complex)

    for i in range(len(t)-1):
    
        #étape 1 :
        ustar[:,i] = (fft.fftshift(fft.fft(u[:,i])))
        
        #étape 2 :
        gstar[:,i] = np.exp((1j * (((2 * np.pi)/ L) *k)**3)*dt) * ustar[:,i]
        
        #étape 3 :
        g[:,i] = (fft.ifft(fft.ifftshift(gstar[:,i])))
        g2 = fft.ifft(fft.ifftshift(((1j * ((2 *np.pi)/ L) * k) * (fft.fftshift(fft.fft(g[:,i]**2))))))
        
        #étape  :
        u[:,i+1] = np.real(g[:,i]-3 * g2 *dt)
        
        #calcul sol analitique 
        sol[:,i] = (c_1/2) * (np.cosh((np.sqrt(c_1)/ 2) * (x - a_1 * L - c_1 * t[i])))**(-2) + (c_2/ 2) * (np.cosh((np.sqrt(c_2)/ 2) * (x - a_2 * L - c_2 * t[i])))**(-2)
        
        
        print("calcul en cours ",i+1,"/",len(t)-1)
        
        if i==(len(t)-2) :
            print("patienter : calcul du graphe en cours...")
    
    return u, sol

sol_n, sol_a = Solve(u,sol)
   
#création du graphe
res = np.linspace(np.min(sol_n), np.max(sol_n), 100) #mettre 50 si calcul trop lent
X,T = np.meshgrid(x, t)
plt.contourf(X, T, sol_n.T, res)  # sol_n.T = sol transposée 
plt.colorbar()
plt.axis('scaled')
plt.ylabel('t')
plt.xlabel('x')
plt.title('Korteweg-de Vries equation for two soliton')
plt.show()



