import matplotlib.pyplot as plt
import numpy as np

SHAPE = 'ellipse'
N = 8
R1 = 3
R2 = 1

if SHAPE == 'ellipse':
    phi = np.random.random(N)*2*np.pi
    x = np.cos(phi) * R1
    y = np.sin(phi) * R2
else:
    x = (np.random.random(N)*2-1)*R1
    y = (np.random.random(N)*2-1)*R2

dist = np.zeros((N, N))  # global distances
for i in range(N):
    dist[i] = np.sqrt((x[i] - x)**2 + (y[i] - y)**2)

NN = min(N-1, 10)
adist = np.zeros((N, NN), int)

for i in range(N):
    Q = np.zeros(N, bool)  # part-of-the-graph
    Q[i] = True
    for nn in range(NN):
        j = np.amin(np.argmin(dist[Q][:, ~Q], 1))
        j = np.where(~Q)[0][j]
        Q[j] = True
        adist[i, nn] = j

plt.scatter(x, y)
plt.scatter(x[0], y[0], color='red')
for s, i in enumerate(adist[0]):
    plt.text(x[i], y[i], '%d(%d)' % (s+1, i))
plt.xlim(-R1, R1)
plt.ylim(-R1, R1)
plt.show()
