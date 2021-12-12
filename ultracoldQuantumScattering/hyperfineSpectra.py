import numpy as np
import matplotlib.pyplot as plt

G_E = 2.002319
G_LI = -0.000447
G_RB = -0.000294
MI_B = 0.5
MI_N = MI_B / 1836.15
F_LI = 152.1 * 10 ** 6 / 4.13 / 10 ** 16
F_RB = 1011.9 * 10 ** 6 / 4.13 / 10 ** 16
B_AU = np.linspace(0, 2000, 100) / 2.35 / 10 ** 9


def create_hamiltonian(size, ind_row, ind_col, li, f):
    H = np.zeros((size, size))
    for i in ind_row:
        for j in ind_col:
            H[i, j] = f * li[i, j] / 2
            H[j, i] = f * li[j, i] / 2
    return H


def simulate(g_n, f, title):
    i = 3 / 2
    s = 3 / 2
    M_i = np.arange(-i, i + 1, 1)
    M_s = np.arange(-s, s + 1, 1)
    I = np.sqrt(i * (i + 1) - M_i[1:] * (M_i[1:] - 1))
    S = np.sqrt(s * (s + 1) - M_s[:-1] * (M_s[:-1] + 1))

    I_block = np.zeros((len(M_i), len(M_i)))
    S_block = np.zeros((len(M_s), len(M_s)))

    np.fill_diagonal(I_block[1:], I)
    np.fill_diagonal(I_block[:, 1:], I)
    np.fill_diagonal(S_block[1:], S)
    np.fill_diagonal(S_block[:, 1:], S)

    # Calculate stripes from block matrices
    stripes = []
    for w in S_block:
        li = np.array([])
        for k in w:
            if len(li) == 0:
                li = k * I_block
            else:
                li = np.hstack((li, k * I_block))
        stripes.append(li)

    # Merge stripes into matrix with the hamiltonian shape
    li = np.array([])
    for st in stripes:
        if len(li) == 0:
            li = st
        else:
            li = np.vstack((li, st))

    spin = np.arange(10, len(M_s) * 10 + 10, 10)
    I_nuclear = np.arange(1, len(M_i) + 1, 1)

    data = []
    for i in spin:
        for j in I_nuclear:
            data.append(i + j)

    data = np.array(data)
    data_pm = data + 9
    data_mp = data - 9
    data = list(data)
    data_pm = list(data_pm)

    up_triangle = np.intersect1d(data, data_pm)
    ind_row = [data.index(d) for i, d in enumerate(up_triangle) if d in data]
    ind_col = [
        data_pm.index(d) for i, d in enumerate(up_triangle) if d in data_pm
    ]

    size = len(M_i) * len(M_s)
    hamiltonian = create_hamiltonian(size, ind_row, ind_col, li, f)

    states = []
    for B0 in B_AU:
        a = 0.5 * G_E * MI_B * B0
        b = 0.5 * g_n * MI_N * B0

        A = a * M_s
        B = b * M_i

        diag = []
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                diag.append(a + b + f * M_s[i] * M_i[j])

        np.fill_diagonal(hamiltonian, np.array(diag))

        w, _ = np.linalg.eig(hamiltonian)
        ind = w.argsort()[::-1]
        w = w[ind]
        states.append(w)

    B_gauss = B_AU * 2.35 * 10 ** 9
    states = np.array(states).T
    states_names = []
    for s in M_s:
        for i in M_i:
            states_names.append(f"|{s}, {i}>")

    fig = plt.figure()
    plt.title(f"Hyperfine spectra of {title}")
    for i, y in enumerate(states):
        plt.plot(B_gauss, y * 4.13 * 10 ** 10, label=states_names[i])
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.xlabel("Magnetic field [G]")
    plt.ylabel("Energy [MHz]")
    plt.legend()
    plt.show(block=False)
    return fig


def main():
    fig1 = simulate(G_LI, F_LI, "Li 6")
    fig2 = simulate(G_RB, F_RB, "Rb 85")
    plt.show()


if __name__ == "__main__":
    main()
