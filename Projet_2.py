## Projet_2 De-Vries equation

import numpy as np
import numpy.fft as fft
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#définition des constantes
L = 50
N = 256
tmax = 70  # faire pour 
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
   
#création du graphe pour sil numérique
res = np.linspace(np.min(sol_n), np.max(sol_n), 100) #mettre 50 si calcul trop lent
X,T = np.meshgrid(x, t)
plt.contourf(X, T, sol_n.T, res)  # sol_n.T = sol transposée 
plt.colorbar()
plt.axis('scaled')
plt.ylabel('t')
plt.xlabel('x')
plt.title('Solution numérique')
plt.show()


##Partie 2
print("calcul du graphe 2 en cours...")

#création du graphe pour sol analytique
res = np.linspace(len(x), len(t), 100) #mettre 50 si calcul trop lent
X,T = np.meshgrid(x, t)
plt.contourf(X, T, sol_a.T, res)  # sol_n.T = sol transposée 
plt.colorbar()
plt.axis('scaled')
plt.ylabel('t')
plt.xlabel('x')
plt.title('solution analityque')
plt.show()
#discuter les résultats + savoir si il y a un periode pour la quel la sol_n aproxime sol_a


#on peut faire une comparaison quantitative en comparant l'amplitude des deux soliton 

delta = sol_a - sol_n

#essai de graphique pour un temps donné mais pas concluant
fig, ax=plt.subplots()
ax.plot(t, delta[40,:], label = "numérique")
#ax.plot(t, sol_a[40,:], label = "analytique")
plt.xlabel("temp")
plt.ylabel("amplitude")
plt.title("comparaison")
plt.legend()
plt.show()

#on peut faire une annimation pour tout t pour mieux visualiser les differances






