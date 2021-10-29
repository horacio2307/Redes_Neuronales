import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Número de muestras 50
#Las curvas son bastante suaves 
#Es un problema de regresión si se aplican más dimensiones
p = 50
x = np.linspace(-5, 5, p).reshape(1,-1)
y = 2 * np.cos(x) + np.sin(3*x) +5

#Dibujar puntos
plt.plot(x,y, 'or')


#---------------------------NO SUPERVISADA
#definir número de 
#Si se tiene la misma cantidad de puntos que de datos se cae en el sobreentrenamiento
k = 30

#Agrupar puntos
model = KMeans(n_clusters = k)
model.fit(x.T)

#Extraer centroides
#clase kMeans
#Cada centroide es para cada neurona, es decir, se tendrán 3 
#cada centroide tratará de acercarse a una porción de la gráfica
c = model.cluster_centers_

#-------------------------- SUPERVISADA
#Calcular la dispersion de cada valor gaussiano
#se genera un arreglo
sigma = (max(c)-min(c))/np.sqrt(2*k)
#lo saco para que sea un float
sigma = sigma[0]

#Calcular salida de cada neurona para cada patron para la neurona j y las entradas i -> matriz G
G = np.zeros((p,k))
#Calcular distancia
#recorre el patron # recorre la neurona
for i in range(p):
    for j in range(k):
        #Para la suma de las distancias
        dist = np.linalg.norm(x[0,i]-c[j],2)
        G[i,j] = np.exp((-1/(sigma**2))* dist**2)
        
#pseudoinversa de G, con esto se entrena la segunda capa
W = np.dot(np.linalg.pinv(G), y.T)

#Propagar la red
p = 200
xnew = x = np.linspace(-5, 5, p).reshape(1, -1)
G = np.zeros((p,k))
for i in range(p):
    for j in range(k):
        #Para la suma de las distancias
        dist = np.linalg.norm(x[0,i]-c[j],2)
        G[i,j] = np.exp((-1/(sigma**2))* dist**2)
ynew = np.dot(G, W)

plt.plot(xnew.T, ynew, '-b')
plt.show()

