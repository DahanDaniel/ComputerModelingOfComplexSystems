from math import ceil

import numpy as np
import matplotlib.pyplot as plt

zero = np.array([
    -1, 1, 1, 1, -1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    -1, 1, 1, 1, -1,
])
one = np.array([
    -1, 1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
])
two = np.array([
    1, 1, 1, -1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, 1, -1,
    -1, 1, 1, -1, -1,
    1, -1, -1, -1, -1,
    1, 1, 1, 1, 1,
])
patterns = [zero, one, two]

w = np.zeros((30, 30))
for miu in patterns:
    matrix = np.outer(miu, miu)
    w += 1/len(patterns)*matrix
w -= np.identity(matrix.shape[0])

def update_network(x, w):
    # Flip spins
    x = np.sign(np.dot(w, x))
    return x

def calc_energy(x, w):
    # Calculate energy
    energy = np.empty(len(x))
    for i in range(len(x)):
        h = 0
        for j in range(len(x)):
            if i != j:
                h += w[i][j]*x[j]
        energy[i] = -1/2*x[i]*h
    return np.sum(energy)

def async_update(x, w):
    # Choose random spin
    randomSpinIdx = np.random.randint(0, len(x))

    # Compare energies before and after flipping the spin
    e1 = calc_energy(x, w)
    x[randomSpinIdx] *= -1
    e2 = calc_energy(x, w)
    if e1 < e2:
        x[randomSpinIdx] *= -1
    return x

def simulate(x, w, N, update_method):
    update_network = update_method

    # Create grid
    no_columns = int(np.sqrt(N))
    no_rows = ceil(N/no_columns)
    fig1, ax1 = plt.subplots(no_rows, no_columns)
    for i in range(ax1.shape[0]):
        for j in range(ax1.shape[1]):
            ax1[i][j].set_axis_off()
    energies = np.zeros(N)

    # Populate the grid with the 0th and 1st iteration of simulation
    energies[0] = calc_energy(x, w)
    ax1[0][0].imshow(x.reshape((6, 5)), interpolation='nearest')
    output = update_network(x, w)
    energies[1] = calc_energy(output, w)
    ax1[0][1].imshow(output.reshape((6, 5)), interpolation='nearest')

    # Populate grid with the rest of iterations
    for j in range(no_columns):
        rang = range(min(no_rows, N-j*no_rows)) if j != 0 else range(2, no_rows)
        for i in rang:
            output = update_network(output, w)
            energies[j*no_rows + i] = calc_energy(output, w)
            ax1[j][i].imshow(output.reshape((6, 5)), interpolation='nearest')
    fig2 = plt.figure()
    plt.plot(np.arange(N), energies)
    plt.show()

inputMatrixZero = np.array([
    -1, 1, 1, 1, -1,
    1, -1, -1, -1, -1,
    1, -1, -1, -1, 1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, 1, -1, -1,
])
inputMatrixOne = np.array([
    1, 1, 1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, 1, -1, -1,
    1, -1, -1, -1, -1,
    1, 1, -1, -1, 1,
])
inputMatrixTwo = np.array([
    1, 1, 1, -1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
])

simulate(inputMatrixZero, w, 4, update_network)
simulate(inputMatrixOne, w, 4, update_network)
simulate(inputMatrixTwo, w, 4, update_network)

# simulate(inputMatrixZero, w, 80, async_update)
# simulate(inputMatrixOne, w, 20, async_update)
# simulate(inputMatrixTwo, w, 80, async_update)
