import collections
from sys import exit
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
U = 1822.888  # atomic units [au]
D_E = 0.0011141  # [au]
R_E = 10.98  # [au]
M1 = 38.963707 * U  # K39 [au]
M2 = 39.963999 * U  # K40 [au]
M3 = 40.961825 * U  # K41 [au]
MIU1 = M1 * M2 / (M1 + M2)  # reduced mass
MIU2 = M1 * M3 / (M1 + M3)  # reduced mass
MIU3 = M2 * M3 / (M2 + M3)  # reduced mass
R_0 = 5
R_N = 20
N = 500


def r_array(r_0, r_n, n):
    return np.array([r_0 + i * (r_n - r_0) / n for i in range(0, n - 1)])


def V_internal(d_e, r_e, r):
    return d_e * ((r_e / r) ** 12 - 2 * (r_e / r) ** 6)


def R_i(i, r_0, r_n, n):
    return r_0 + i * (r_n - r_0) / n


def T_ii(i, n, miu, r_0, r_n):
    a = 1 / 3 * (2 * n ** 2 + 1)
    b = np.sin(i * np.pi / n)
    c = 4 * miu * (r_n - r_0) ** 2
    return np.pi ** 2 * (a - np.power(b, -2)) / c


def T_ij(i, j, n, miu, r_0, r_n):
    a = np.pi ** 2 * (-1) ** (-i - j)
    b = np.sin((i - j) * np.pi / (2 * n))
    c = np.sin((i + j) * np.pi / (2 * n))
    d = 4 * miu * (r_n - r_0) ** 2
    return a * (np.power(b, -2) - np.power(c, -2)) / d


def V_ij(i, J, miu):
    return J * (J + 1) / (2 * miu * R_i(i, R_0, R_N, N) ** 2) + V_internal(
        D_E, R_E, R_i(i, R_0, R_N, N)
    )


def V_harmonic(k, r):
    return k * (r - R_E) ** 2 / 2


def V_harmonic_displaced(k, r, difference):
    return k * (r - R_E) ** 2 / 2 - difference


def V_ij_and_V_harmonic(i, J, miu, difference, k):
    return J * (J + 1) / (
        2 * miu * R_i(i, R_0, R_N, N) ** 2
    ) + V_harmonic_displaced(k, R_i(i, R_0, R_N, N), difference)


def create_matrix_internal(n, miu, r_0, r_n, J, difference=None, k=None):
    matrix = np.zeros((n - 1, n - 1))
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                matrix[i - 1][j - 1] = T_ii(i, n, miu, r_0, r_n) + V_ij(
                    i, J, miu
                )
            else:
                matrix[i - 1][j - 1] = T_ij(i, j, n, miu, r_0, r_n)
    return matrix


def create_matrix_harmonic(n, miu, r_0, r_n, J, difference, k):
    matrix = np.zeros((n - 1, n - 1))
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                matrix[i - 1][j - 1] = T_ii(
                    i, n, miu, r_0, r_n
                ) + V_ij_and_V_harmonic(i, J, miu, difference, k)
            else:
                matrix[i - 1][j - 1] = T_ij(i, j, n, miu, r_0, r_n)
    return matrix


def get_eigen(
    n, miu, r_0, r_n, J, create_matrix_func, difference=None, k=None
):
    # Hamiltonian and its eigenvalues
    hamiltonian = create_matrix_func(n, miu, r_0, r_n, J, difference, k)
    eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
    eigenvectors = np.transpose(eigenvectors)
    eigenvectors_and_vals_df = pd.DataFrame(columns=["Energy", "Vector"])
    eigenvectors_and_vals_df["Energy"] = eigenvalues
    eigenvectors_and_vals_df["Vector"] = [
        eigenvectors[i] for i in range(len(eigenvalues))
    ]
    eigenvectors_and_vals_df = eigenvectors_and_vals_df.sort_values(
        "Energy", ascending=True
    ).reset_index()
    return eigenvectors_and_vals_df


def second_derivative(r_e, d_e):
    a = d_e * r_e ** 12
    b = 2 * d_e * r_e ** 6
    return (156 * a / r_e ** 14) - (42 * b / r_e ** 8)


def main():
    # Distances array
    R_array = r_array(R_0, R_N, N)

    eigenvectors_and_vals_df = get_eigen(
        N, MIU2, R_0, R_N, 1, create_matrix_internal
    )  # J = 1

    # Plot first 4 eigenvectors
    NUM_VECTORS = 4
    fig, axs = plt.subplots(NUM_VECTORS)
    fig.suptitle("Eigenvectors in internal potential")
    for i in range(NUM_VECTORS):
        energy = eigenvectors_and_vals_df.loc[i, "Energy"]
        vector = eigenvectors_and_vals_df.loc[i, "Vector"]
        axs[i].plot(R_array, vector)
    plt.show(block=False)

    # Plot energies in Lenard Jones potential
    fig = plt.figure()
    fig.suptitle("Energy in Lenard Jones Potential")
    plt.plot(eigenvectors_and_vals_df["Energy"][:40])
    plt.show(block=False)

    # Eigenvectors for different J
    fig = plt.figure()
    fig.suptitle("Eigenvectors for different J")
    for J in [0, 1, 20, 40, 60, 80, 100]:
        eigenvectors_and_vals_df = get_eigen(
            N, MIU2, R_0, R_N, J, create_matrix_internal
        )
        vector = eigenvectors_and_vals_df.loc[0, "Vector"]
        plt.plot(R_array, np.abs(vector), label=f"{J}")
    plt.legend()
    plt.show(block=False)

    # Eigenvectors for different masses
    masses = [MIU1, MIU2, MIU3, M1 / 2, M2 / 2, M3 / 2]
    fig = plt.figure()
    fig.suptitle("Eigenvectors for different masses")
    for mass in masses:
        eigenvectors_and_vals_df = get_eigen(
            N, mass, R_0, R_N, 1, create_matrix_internal
        )
        vector = eigenvectors_and_vals_df.loc[0, "Vector"]
        plt.plot(R_array, np.abs(vector), label=f"{mass}")
    plt.legend()
    plt.show(block=False)

    # Plot internal and harmonic potential
    k = second_derivative(R_E, D_E)
    # Compare potentials
    V_harmonic_min = V_harmonic(k, R_E)
    V_internal_min = -0.001114  # From plot
    difference = V_harmonic_min - V_internal_min
    # Plot
    fig = plt.figure()
    fig.suptitle("Internal and harmonic potential")
    V_int_array = np.array(
        [V_internal(D_E, R_E, r) for r in r_array(9.47, 13, N)]
    )
    V_harm_array = np.array([V_harmonic(k, r) for r in r_array(9.47, 13, N)])
    plt.plot(r_array(9.47, 13, N), V_int_array, label="Internal")
    plt.plot(r_array(9.47, 13, N), V_harm_array - difference, label="Harmonic")
    plt.show(block=False)

    eigenvectors_and_vals_df = get_eigen(
        N, MIU2, R_0, R_N, 1, create_matrix_harmonic, difference, k
    )

    # Plot first 4 eigenvectors for harmonic potential
    NUM_VECTORS = 4
    fig, axs = plt.subplots(NUM_VECTORS)
    fig.suptitle("Eigenvectors in harmonic potential")
    for i in range(NUM_VECTORS):
        energy = eigenvectors_and_vals_df.loc[i, "Energy"]
        vector = eigenvectors_and_vals_df.loc[i, "Vector"]
        axs[i].plot(R_array, vector)
    plt.show(block=False)

    plt.show()


if __name__ == "__main__":
    main()
