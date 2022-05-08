## Projet_2 De-Vries equation

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

#Constantes du problème
L = 50
N = 256
dt = 0.00004
tmax = 200
j = complex(0,1)
x = np.linspace(0,L,N)
t = np.linspace(0,tmax,N)

c1 = 0.75
a1 = 0.33
c2 = 0.4
a2 = 0.65

#condition initialle 
u_0 = (c1/2)*np.cosh((np.sqrt(c1)/2)*(x - a1*L))**(-2) * (c2/2)*np.cosh((np.sqrt(c2)/2)*(x - a2*L))**(-2)
u = np.zeros((len(t),len(x)))
u[0] = u_0

def solve(u):
    ustar = np.zeros((len(t),len(x)))
    gstar = np.zeros((len(t),len(x)))
    g = np.zeros((len(t),len(x)))
    g2 = np.zeros((len(t),len(x)))


    for i in range(len(t)-1):
        
        #étape 1
        ustar[i] = fft.fft(u[i])
        
        #étape 2
        gstar[i] = ustar[i] * np.exp(j*((2*np.pi*i)/L)**3 * dt) 
        
        #étape 3
        g[i] = fft.ifft(gstar[i])
        g2[i] = fft.ifft((j*(2*np.pi)/L)*i*fft.fft(g[i]**2))
        
        #étape 4
        u[i+1] = g[i] - 3*g2[i]*dt
        
        
    return u


sol = solve(u) 


#sol_x = sol[:]
#sol_t = sol[:]     

plt.pcolormesh(x, sol,sol ,shading='gouraud')
plt.ylabel('t')
plt.xlabel('x')
#plt.title('Spectrogramme du Cri Whilhem')
plt.colorbar()
plt.show()



#plt.figure(figsize=(12, 3))
#plt.subplot()
#plt.imshow(sol)
#plt.colorbar()
#plt.show()

# création d'une boucle for pour afficher tous les graphiques

#plt.figure(figsize=(12, 8))
#for i in range(N-1):
    #plt.subplot(N//2, N//2, i+1)
    #plt.scatter(sol[:, 0], sol[:, i]) # affiche la variable i en fonction de la variable 0
    #plt.xlabel('0')
    #plt.ylabel(i)
    #plt.colorbar(ticks=list(np.unique(t)))
    #print("calcul en cours :", i,"/",N)
#plt.show()



#Affichage de la surface
ax = plt.axes(projection='3d')
ax.plot_surface(t, x,sol, cmap='plasma')

#X = np.linspace(0, 5, 100)
#Y = np.linspace(0, 5, 100)
#X, Y = np.meshgrid(X, Y)
#Z = f(X, Y) 
#plt.contour(x, t, sol, levels=40)



#plt.figure(figsize=(12, 3))
 

 
#plt.subplot(131)
#plt.imshow(X)
 

 
#plt.subplot(132)
#plt.imshow(np.corrcoef(X.T, y))
 

#plt.subplot()
#plt.imshow(sol)
#plt.colorbar()
