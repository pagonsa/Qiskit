import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit_aer import Aer
from qiskit import assemble
from qiskit.visualization import plot_histogram

sv_sim = Aer.get_backend('aer_simulator')

grover = qiskit.QuantumCircuit(3, name='grov')
grover.mcp(np.pi, [0, 1], 2)    #oraculo
grover.h([0, 1, 2])
grover.x([0, 1, 2])
grover.mcp(np.pi, [0, 1], 2)
grover.x([0, 1, 2])
grover.h([0, 1, 2])

grov = grover.to_gate()

qc = qiskit.QuantumCircuit()
qr = qiskit.QuantumRegister(3)
cr = qiskit.ClassicalRegister(3)
qc.add_register(qr, cr)
qc.h([qr[0], qr[1], qr[2]])     #inicializar sistema
qc.append(grov, [qr[0], qr[1], qr[2]])

qc.draw(output="mpl")
grover.draw(output="mpl")

qc.save_statevector()
qobj = assemble(qc)
job = sv_sim.run(qobj)
ket = job.result().get_statevector()
for amplitude in ket:
    print(amplitude)
hist = job.result().get_counts()
print(hist)
plot_histogram(hist)
plt.show()

