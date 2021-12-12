from sys import exit

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

POINTS_DENSITY = 100  # points per width unit
MARGIN = 5  # 1 is well only, the rest are walls
N = MARGIN * POINTS_DENSITY  # mesh points
ENERGY_GAP = 0.5  # eV
WIDTH = 40  # Å
DX = MARGIN * WIDTH / (N - 1)

CONST1 = 3.80996 / 0.3  # eV * Å ** 2


def create_hamiltonian(potential_arr):
    n = len(potential_arr)
    hamiltonian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                hamiltonian[i][j] = CONST1 * 2 / (DX ** 2) + potential_arr[i]
            elif j == i + 1 or j == i - 1:
                hamiltonian[i][j] = -1 * CONST1 / (DX ** 2)
    return hamiltonian


def create_square_well(energy_gap, points_density) -> np.ndarray:
    arr = np.zeros(MARGIN * points_density)
    arr[2 * points_density : 3 * points_density] -= energy_gap
    return arr


def create_parabolic_well(energy_gap, points_density) -> np.ndarray:
    # Parabola part
    x = np.linspace(-1, 1, points_density)
    y = 0.5 * x ** 2 - energy_gap

    arr = np.zeros(MARGIN * points_density)
    arr[2 * points_density : 3 * points_density] = y
    return arr


def simulate(potential_arr):
    # Initialize scenario
    hamiltonian = create_hamiltonian(potential_arr)

    # Find eigenvalues and eigenvectors
    w, v = LA.eig(hamiltonian)  # already normalized
    eigenvalues = w[w < 0]  # only negative energy solutions
    eigenvectors = v.T[
        w < 0
    ]  # only vectors corresponding to negative energies

    # Sort results from lowest energy to highest
    sort_indices = eigenvalues.argsort()
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[sort_indices]

    # Plot solutions
    fig = plt.figure()
    distance = np.linspace(0, MARGIN * WIDTH, N)
    plt.plot(distance, potential_arr)
    for idx, eigenvector in enumerate(eigenvectors):
        plt.plot(distance, eigenvector + eigenvalues[idx])
        plt.plot(distance, np.ones(N) * eigenvalues[idx], linestyle="dashed")
    fig.supxlabel("Distance [Å]")
    fig.supylabel("Energy [eV]")
    plt.show(block=False)
    return fig


def main():
    square_well_arr = create_square_well(ENERGY_GAP, POINTS_DENSITY)
    parabolic_well_arr = create_parabolic_well(ENERGY_GAP, POINTS_DENSITY)

    fig1 = simulate(square_well_arr)
    fig1.suptitle("Square well")
    fig2 = simulate(parabolic_well_arr)
    fig2.suptitle("Parabolic well")
    plt.show()


if __name__ == "__main__":
    main()
