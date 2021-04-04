import numpy as np

class Neuronal_Network:

    def __init__(self):
        self.weights = 10*np.random.random(3)
        self.data = np.array([[-1,-1,-1],[-1,1,-1],[1,-1,-1],[1,1,-1]])
        self.target = np.array([-1,-1,-1,1]) #Para and
        
        

    def prediction(self,i):
        a = np.dot(self.weights,self.data[i])


        a = 1 / (1 + np.exp(-a))
        if a < 0.5:
            return -1

        else:
            return 1
    
    def test(self,inputs):
        a = np.dot(self.weights,inputs)
        a = 1 / (1 + np.exp(-a))
        if a < 0.5:
            return 0

        else:
            return 1


    def train(self,i,inputs): #Inputs should be 1x3 [x,y,-1]
        self.weights = self.weights + self.target[i]*inputs

def pedir():
    x = [0,0,-1]
    for i in range(2):
        x[i] = int(input("Ingrese la entrada {}: ".format(i+1)))
        if x[i] <= 0:
            x[i] = -1
        else:
            x[i] = 1
    
    return x

Neurona = Neuronal_Network()
answers = np.zeros(4)
i = 0
while i < 4 :
    if Neurona.prediction(i) != Neurona.target[i]:
        Neurona.train(i,Neurona.data[i])
        i=0
    else:
        i+=1
print("Neurona entrenada :)")
print("Valores de los pesos {}".format(Neurona.weights))

entrada = input("1--> Teste\n2-->Salir\n")
while entrada == "1":
    for y in range (4):
        lista = pedir()
        print(Neurona.test(lista))
    entrada = input("1--> Teste\n2-->Salir\n")