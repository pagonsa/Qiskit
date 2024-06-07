import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer

import qiskit
from qiskit.algorithms import EstimationProblem
from qiskit.algorithms import IterativeAmplitudeEstimation
from qiskit import BasicAer
from qiskit.utils import QuantumInstance

sv_sim = Aer.get_backend('aer_simulator')


sup = 10                      #limites integracion
inf = 0
T = sup-inf                 #intervalo de la serie
N = 10                        #cantidad coeficientes de Fourier
q = 8                      #numero de qubits
s = 100000                  #shots integral cuantica
tam = 100000                 #valores del array x para sacar serie de fourier




def f(x):                   #funcion a integrar
    return 2*x


x = np.linspace(inf, sup, tam)            #array de valores de x para obtener coeficientes fourier
dx = x[1] - x[0]

a0 = 1/T*np.sum(f(x)*dx)                        #constante de la suma

a = np.zeros(N)                                 #coef coseno
for i in range(0, N):
    a[i] = 2/T*np.sum(f(x) * np.cos(2 * np.pi * (i+1) * x / T) * dx)     #para que sume en x el unico array tiene que ser x

b = np.zeros(N)                                 #coef seno
for i in range(0, N):
    b[i] = 2/T*np.sum(f(x)*np.sin(2*np.pi*(i+1)*x/T)*dx)


def P(qc, qr, n):                   #distribucion uniforme (hadamards)
    qc.h(qr[0:n])


def R(qc, qr, max, min, n):                                 #quantum circuit y register, intervalo seno, n qubits (sin contar auxiliar)
    qc.ry((max - min) / 2 ** n + 2 * min, qr[n])            # primera rotacion sin control
    for i in range(0, n):                                   # n rotaciones controladas
        qc.cry((max - min) / 2 ** (n - i - 1), qr[i], qr[n])

def Rinv(qc, qr, max, min, n):
    for i in range(0, n)[::-1]:                                       # n rotaciones controladas
        qc.cry(-(max - min) / 2 ** (n - i - 1), qr[i], qr[n])
    qc.ry(-(max - min) / 2 ** n - 2 * min, qr[n])                # primera rotacion sin control (ahora ultima)

def A_operator(max, min, n):                    #A=PR
    qr = qiskit.QuantumRegister(n+1)
    A = qiskit.QuantumCircuit(qr)               #crea circuito A con su registro
    P(A, qr, n)                                 #Aplica P y R
    R(A, qr, max, min, n)
    return A

def Q_operator(max, min, n):
    qr = qiskit.QuantumRegister(n+1)
    Q = qiskit.QuantumCircuit(qr)
    Q.z(qr[n])                              #z sobre ultimo qubit ("oraculo" entre bueno y malo)
    Rinv(Q, qr, max, min, n)                #Rinv-P es Ainv
    P(Q, qr, n)
    Q.x(qr)                                 #Este es el Z_OR como en Grover
    Q.mcp(np.pi, qr[1:], 0)
    Q.x(qr)
    P(Q, qr, n)                             #P-R es A
    R(Q, qr, max, min, n)
    return Q


def IQAE(max, min, n, s):
    A=A_operator(max, min, n)
    Q=Q_operator(max, min, n)
    problem = EstimationProblem(
    state_preparation=A,  # A operator
    grover_operator=Q,  # Q operator
    objective_qubits=[n],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
    )

    quantum_instance = QuantumInstance(sv_sim, shots=s)
    iae = IterativeAmplitudeEstimation(
    epsilon_target=0.005,  # target accuracy
    alpha=0.05,  # width of the confidence interval
    quantum_instance=quantum_instance,
    )
    iae_result = iae.estimate(problem)
    return(iae_result.estimation)

def sin2(a, f, max, min, n):            #Integra un seno cuadrado (SIGUE CON LA MEDIDA A PELO)
    maxd = a*max + f                    #Redefinir maximos y minimos para quitar a y f
    mind = a*min + f
    a_est = IQAE(maxd, mind, n, s)
    return a_est * (max - min)

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

integral = intcos(a, sup, inf) + intsin(b, sup, inf) + a0 * (sup - inf)
print(integral)
