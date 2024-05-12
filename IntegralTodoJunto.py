import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

sv_sim = Aer.get_backend('aer_simulator')


max=10                      #limites integracion
min=0.001
T = max-min                 #intervalo de la serie
N=10                        #cantidad coeficientes de Fourier
n = np.arange(1, N+1, 1)    #array 1,2,3,4 para multiplicar el argumento de cada coseno/seno
q = 8                      #numero de qubits
s = 10000                  #shots integral cuantica
tam = 10000                 #valores del array x para sacar serie de fourier


def f(x):                   #funcion a integrar
    return np.log(x) ** 2


x = np.linspace(min, max, tam)            #array de valores de x para obtener coeficientes fourier
dx = x[1] - x[0]

a0 = 1/T*np.sum(f(x)*dx)                        #constante de la suma

a = np.zeros(N)                                 #coef coseno
for i in range(0, N):
    a[i] = 2/T*np.sum(f(x) * np.cos(2 * np.pi * n[i] * x / T) * dx)     #para que sume en x el unico array tiene que ser x

b = np.zeros(N)                                 #coef seno
for i in range(0, N):
    b[i] = 2/T*np.sum(f(x)*np.sin(2*np.pi*n[i]*x/T)*dx)



def sin2(a, f, max, min, n):            #Integra un seno cuadrado
    maxd = a*max + f                    #Redefinir maximos y minimos para quitar a y f
    mind = a*min + f

    qc=qiskit.QuantumCircuit()
    qr=qiskit.QuantumRegister(n+1)            #n qubits mas auxiliar
    cr=qiskit.ClassicalRegister(1)
    qc.add_register(qr,cr)
    for i in range(0,n):                      #Aplica hadamard a todos para tener p(x) uniforme
        qc.h(qr[i])
    qc.ry((maxd-mind)/2**n+2*mind, qr[n])     #primera rotacion sin control
    for i in range(0, n):                     #n rotaciones controladas
        qc.cry((maxd-mind)/2**(n-i-1), qr[i], qr[n])
    qc.measure(qr[n],cr[0])                   #Al medir el qubit auxiliar nos da el promedio de la funcion, multiplicado por max-min es el valor de la integral
    qc.save_statevector()
    new_circuit = qiskit.transpile(qc, sv_sim)
    job = sv_sim.run(new_circuit, shots=s)
    hist = job.result().get_counts()
    return hist['1']*(max-min)/s              #el promedio por el rango de integracion es la integral


#tenemos cos(2npi x / T) que pasado a sin cuadrado seria (con T = max-min)
#1-2sin^2(npi x /T) por lo que el coeficiente de la x seria npi/T y f=0. n=i+1 porque empieza en 0
def intcos(Fcos, max, min):
    C = np.size(Fcos)
    res = 0
    for i in range(0, C):
        res = res + Fcos[i]*((max - min) - 2*sin2((i+1)*np.pi/(max-min), 0, max, min, q))
    return res

#tenemos sin(2npi x / T) que pasado a sin cuadrado seria (con T = max-min)
#1-2sin^2(npi x /T-pi/4) por lo que el coeficiente de la x seria npi/T y f=-pi/4. n=i+1 porque empieza en 0
def intsin(Fsin, max, min):
    C = np.size(Fsin)
    res = 0
    for i in range(0, C):
        res = res + Fsin[i]*((max-min) - 2*sin2((i+1)*np.pi/(max-min), -np.pi/4, max, min, q))
    return res

integral = intcos(a, max, min) + intsin(b, max, min) + a0 *(max - min)
print(integral)
