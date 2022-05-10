## Projet_2 De-Vries equation

import numpy as np
import numpy.fft as fft
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#définition des constantes
L = 50
N = 256
tmax = 50  # faire pour 
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
ua_1 = np.zeros((len(x), len(t)))
u_1 = np.zeros((len(x), len(t)))

#condition initiale
u_0 = (c_1/2) * (np.cosh((np.sqrt(c_1)/ 2) * (x - a_1 *L)))**(-2) + (c_2/ 2) * (np.cosh((np.sqrt(c_2)/ 2) * (x - a_2 * L)))**(-2)
u_1[:,0] = u_0

def Solve1(u, ua):
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
        
        #calcul sol analitique pour 2 soliton
        ua_1[:,i] = (c_1/2) * (np.cosh((np.sqrt(c_1)/ 2) * (x - a_1 * L - c_1 * t[i])))**(-2) + (c_2/ 2) * (np.cosh((np.sqrt(c_2)/ 2) * (x - a_2 * L - c_2 * t[i])))**(-2)
        
        print("calcul en cours ",i+1,"/",len(t)-1)
    
    return u, ua

#calcul des solutions
sol_n, sol_a = Solve1(u_1,ua_1)
  
print("calcul du graphe 1 en cours...")
#création du graphe pour sol numérique
res = np.linspace(np.min(sol_n), np.max(sol_n), 100) #mettre 50 si calcul trop lent
X,T = np.meshgrid(x, t)
plt.contourf(X, T, sol_n.T, res)  # sol_n.T = sol transposée 
plt.colorbar().set_label('amplitude', labelpad=-40, y=1.07, rotation=0)
plt.axis('scaled')
plt.ylabel('t')
plt.xlabel('x')
plt.title('Solution numérique')
plt.show()



######################################################################
######################################################################

##Partie 2


print("calcul du graphe 2 en cours...")

#création du graphe pour sol analytique
res = np.linspace(np.min(sol_a), np.max(sol_a), 100) #mettre 50 si calcul trop lent
X,T = np.meshgrid(x, t)
plt.contourf(X, T, sol_a.T, res)  # sol_n.T = sol transposée 
plt.colorbar().set_label('amplitude', labelpad=-40, y=1.07, rotation=0)
plt.axis('scaled')
plt.ylabel('t')
plt.xlabel('x')
plt.title('solution analityque')
plt.show()

#discuter les résultats + savoir si il y a un periode pour la quel la sol_n aproxime sol_a


#on peut faire une comparaison quantitative en comparant l'amplitude des deux soliton 
diff_1 = sol_a - sol_n

#essai de graphique pour un temps donné mais pas concluant
#fig, ax=plt.subplots()
#ax.plot(t, diff_1[40,:], label = "comparaison")
#ax.plot(t, sol_n[40,:], label = "numérique")
#ax.plot(t, sol_a[40,:], label = "analytique")
#ax2 = ax.twinx()
#ax2.set_ylabel('Amplitude') 
#ax2.plot(x, sol_n[:,40])
#ax2.plot(x, sol_a[:,40])
#plt.xlabel("temp")
#plt.ylabel("amplitude")
#plt.title("je sais pas trop")
#ax.grid(True)
#plt.legend()
#plt.show()

#autre essai
#ax = plt.axes(projection='3d')
#ax.plot_surface(t,sol_n,sol_a, cmap='plasma')
#ax.legend(['temps','sol numérique','sol analytique'])
#plt.show()


#on peut faire une annimation pour tout t pour mieux visualiser les differances

fig, ax = plt.subplots()
graph_n, = ax.plot(x, sol_n[:,0])
graph_a, = ax.plot(x, sol_a[:,0])

#initialisation du l'animation
def init():
        graph_n.set_data(x, sol_n[:,0])
        graph_a.set_data(x, sol_a[:,0])
        return graph_n, graph_a

#déf pour l,'animation
def anim1(frame):
    y_n = sol_n[:,frame]
    y_a = sol_a[:,frame]
    graph_n.set_data((x, y_n))
    graph_a.set_data((x,y_a))
    return graph_n, graph_a


frame = np.arange(0, Nt, 250)

anim1 = FuncAnimation(fig, anim1(frame), init_func = init, interval = 1.3)
plt.axis([0, 50, 0, 0.6])
plt.xticks(np.arange(0, 50, 5))
plt.yticks((np.arange(0, 0.6, 0.06)))
plt.title("Comparaison evolution amplitude", fontsize = 15)
plt.xlabel("x", fontsize = 10)
plt.ylabel("amplitude", fontsize = 10)
plt.legend(["numerique", "analytique"], fontsize = 10, markerscale = 10)
ax.grid(True)
anim1.save('Comparaison sol_a et sol_n')
plt.show()




### comparaison très qualitative pour 1 soliton

#nouvelle condition initiale
u_02 = (c_1/2) * (np.cosh((np.sqrt(c_1)/ 2) * (x - a_1 *L)))**(-2)
ua_2 = np.zeros((len(x), len(t)))
u_2 = np.zeros((len(x), len(t)))
u_2[:,0] = u_02 

 
#déf calcul pour 1 soliton 
def Solve2(u, ua):
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
        
        #calcul sol analitique pour 1 soliton
        ua[:,i] = (c_1/2) * (np.cosh((np.sqrt(c_1)/ 2) * (x - a_1 * L - c_1 * t[i])))**(-2)
        
        print("calcul en cours ",i+1,"/",len(t)-1)
    
    return u, ua

sol_n2, sol_a2 = Solve2(u_2,ua_2)

#graphe pour les sol pour 1 soliton

print("calcul graphe 4 en cours...")
#solution numérique
res = np.linspace(np.min(sol_n2), np.max(sol_n2), 100) #mettre 50 si calcul trop lent
X,T = np.meshgrid(x, t)
plt.contourf(X, T, sol_n2.T, res)  # sol_n.T = sol transposée 
plt.colorbar().set_label('amplitude', labelpad=-40, y=1.07, rotation=0)
plt.axis('scaled')
plt.ylabel('t')
plt.xlabel('x')
plt.title('solution numérique pour 1 soliton')
plt.show()

print("calcul graphe 5 en cours...")
#solution analytique
res = np.linspace(np.min(sol_a2), np.max(sol_a2), 100) #mettre 50 si calcul trop lent
X,T = np.meshgrid(x, t)
plt.contourf(X, T, sol_a2.T, res)  # sol_n.T = sol transposée 
plt.colorbar().set_label('amplitude', labelpad=-40, y=1.07, rotation=0)
plt.axis('scaled')
plt.ylabel('t')
plt.xlabel('x')
plt.title('solution analityque pour 1 soliton')
plt.show()

#Comparaison entre les deux solutions
diff_2 = sol_a2 - sol_n2


#visualisation diférence
print("calcul graphe 6 en cours...")
res = np.linspace(np.min(diff_2), np.max(diff_2), 100) #mettre 50 si calcul trop lent
X,T = np.meshgrid(x, t)
plt.contourf(X, T, diff_2.T, res)   
plt.colorbar().set_label('amplitude', labelpad=-40, y=1.07, rotation=0)
plt.axis('scaled')
plt.ylabel('t')
plt.xlabel('x')
plt.title('différence entre sol_n2 et sol_a2')
plt.show()


#on crée une annimation pour visualiser cette differance au cours du temps

fig, ax = plt.subplots()
graph_diff, = ax.plot(x, diff_2[:,0])

#initialisation 
def ini2():
    graph_diff.set_data(x, diff_2[:,0])
    return graph_diff

#def pour l'animation 
def anim2(frame):
        y = diff_2[:,frame]
        graph_diff.set_data((x, y))
        return graph_diff

frame2 = np.arange(0, Nt, 250)

#lancement de l'animation

anim2 = FuncAnimation(fig, anim2(frame2), init_func = init, interval = 1.3)
plt.axis([0, 50, -0.01, 0.01])
plt.xticks(np.arange(0, 50, 5))
plt.yticks((np.arange(-0.01, 0.01, 0.001)))
plt.title(" différence des amplitude pendant le temps ")
plt.xlabel(" x ")
plt.ylabel("amplitude")
ax.grid(True)
anim2.save('Comparaison sol_a et sol_n pour 1 soliton')
plt.show()

