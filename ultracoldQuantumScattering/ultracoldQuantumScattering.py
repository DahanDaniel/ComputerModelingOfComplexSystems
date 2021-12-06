import numpy as np
import matplotlib.pyplot as plt
import scipy.special

# Constants
U = 1822.888  # atomic units [au]
D_E = 0.0011141  # [au]
R_E = 10.98  # [au]
M1 = 38.963707 * U  # K39 [au]
M2 = 39.963999 * U  # K40 [au]
M3 = 40.961825 * U  # K41 [au]
MIU = M1 * M3 / (M1 + M3)  # reduced mass
N = 500
H = 0.02  # spatial step
h = H ** 2 / 12

E = np.logspace(-11, -6, base=10, num=N) * 3.1667909  # 10 [uK] - 1 [K]
E_K = E * 10 ** 6 / 3.1667909  # Energy in Hartree
R = np.arange(0.863 * R_E, 20 * R_E, H)


def v_tot(R, l):
    r = (R_E / R) ** 6
    return D_E * r * (r - 2) + l * (l + 1) / (
        2 * MIU * R ** 2
    )  # LJ + centrifrugal V


def F(F_prev, k, i):
    fi_n_1 = 2 * (1 - 5 * h * k[i]) - (1 + h * k[i - 1]) / F_prev
    fi_n = 1 + h * k[i + 1]
    return fi_n_1 / fi_n


def calculate_sigma_l(E, l):
    k_bis = np.sqrt(2 * MIU * E)  # potential in infinity
    k = 2 * MIU * (E - v_tot(R, l))  # k ** 2

    F1 = 2 * (1 - 5 * h * k[1]) / (1 + h * k[2])
    for i in range(2, k.size - 1):
        F1 = F(F1, k, i)

    A1 = k_bis * R[-1]  # for fi_n_1
    A0 = k_bis * R[-2]  # for fi_n
    K = (A1 * scipy.special.jv(l, A1) - A0 * F1 * scipy.special.jv(l, A0)) / (
        A0 * F1 * scipy.special.yn(l, A0) - A1 * scipy.special.yn(l, A1)
    )

    S = (1 + 1j * K) / (1 - 1j * K)

    return np.pi / k_bis ** 2 * np.abs(1 - S) ** 2


def main():
    sigma_tot = np.zeros(len(E))
    fig, a = plt.subplots()
    for l in range(5):
        sigma_el = []
        for e in E:
            sigma_el.append(calculate_sigma_l(e, l))
        a.plot(E_K, sigma_el, label="l={0}".format(l), linewidth=1)
        sigma_tot = sigma_tot + np.array(sigma_el)

    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Temperature [K]")
    a.set_ylabel("Elastic cross section")
    a.set_ylim(bottom=10)
    a.plot(E_K, sigma_tot, label="total")
    a.legend()
    plt.show()


if __name__ == "__main__":
    main()
