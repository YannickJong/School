import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


bin_edges = np.arange(30, 71, 1)

# ----- Boxplot -----

# with open("1photon.txt", "r") as f:
#     lengths1 = np.array([int(line) for line in f.readlines()])
#
# with open("10photons.txt", "r") as f:
#     lengths10 = np.array([int(line) for line in f.readlines()])
#
# with open("100photons.txt", "r") as f:
#     lengths100 = np.array([int(line) for line in f.readlines()])
#
#
#
# hist1, _ = np.histogram(lengths1, bins=bin_edges, density=True)
# hist10, _ = np.histogram(lengths10, bins=bin_edges, density=True)
# hist100, _ = np.histogram(lengths100, bins=bin_edges, density=True)
#
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#
# popt1, pcov1 = curve_fit(func, bin_centers, hist1, p0=[1, np.mean(lengths1), np.std(lengths1)])
# popt10, pcov10 = curve_fit(func, bin_centers, hist10, p0=[1, np.mean(lengths10), np.std(lengths10)])
# popt100, pcov100 = curve_fit(func, bin_centers, hist100, p0=[1, np.mean(lengths100), np.std(lengths100)])
#
# means = np.array([popt1[1], popt10[1], popt100[1]])
# stds = np.array([popt1[2], popt10[2], popt100[2]])
#
# plt.boxplot([lengths1, lengths10, lengths100], widths=0.5, showfliers=False, labels=["1 photon", "10 photons", "100 photons"])
# plt.grid()
# plt.ylabel("Length of shared key [bits]")
# plt.savefig("BB84_boxplot.pdf", bbox_inches='tight', dpi=300)
# plt.show()



# ----- Histogram -----

with open("100photons.txt", "r") as f:
    lengths = np.array([int(line) for line in f.readlines()])

hist, _ = np.histogram(lengths, bins=bin_edges, density=True)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
popt, pcov = curve_fit(func, bin_centers, hist, p0=[1, np.mean(lengths), np.std(lengths)])

plt.hist(lengths, bins=bin_edges, density=True, alpha=0.5, label='Histogram')
plt.plot(bin_centers, func(bin_centers, *popt), 'r--', label='Fitted Curve')
plt.legend()
plt.xlabel("Length of shared key [bits]")
plt.ylabel("Probability density [-]")
plt.grid()
plt.savefig("BB84_histogram100photons.pdf", bbox_inches='tight', dpi=300)
print(popt, np.sqrt(np.diag(pcov)))
plt.show()
