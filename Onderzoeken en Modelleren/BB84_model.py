import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(61789324)

class BB84:
    """A class for the BB84 protocol"""
    def __init__(self, n, bitlength):
        """Initializes the BB84 protocol with n photons and a bitlength of
         bitlength"""
        self.n = n
        self.bitlength = bitlength
        self.bit_map = {("+", 0): 0, ("+", 1): 90, ("x", 0): -45, ("x", 1): 45}

    def generate_bases(self):
        """Generates a random set of bases"""
        return np.random.choice(["x", "+"], size=self.bitlength)

    def generate_bits(self):
        """Generates a random set of bits"""
        return np.random.choice([0, 1], size=self.bitlength)

    def calculate_angles(self, bases, bits):
        """Calculates the angles for the bases and bits"""
        return np.array([self.bit_map[(bases[i], bits[i])] for i in range(self.bitlength)])

    def polarizer(self, received_photons, angle):
        """Calculates the polarized photons"""
        M = np.matrix([[np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))],
                       [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))]])
        return np.full((self.n, 2), np.matmul(M, received_photons[0]))

    def measurement(self, received_photons):
        """Simulates the effect of a polarization beam splitter
         on the photons"""
        H = []
        V = []
        for photon in received_photons:
            if np.random.choice(["H", "V"], p=[photon[0]**2, photon[1]**2]) == "H":
                H.append(photon)
            else:
                V.append(photon)
        return 0 if len(H) > len(V) else 1

    def send_photons(self, angle):
        """Sends the photons to Bob"""
        return np.full((self.n, 2), [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])

    def generate_key(self, eve=False):
        """Generates the key for Alice and Bob"""
        # Alice generates a random set of bases and bits, and finds the
        # corresponding angles
        bases_a = self.generate_bases()
        bits_a = self.generate_bits()
        theta_list = self.calculate_angles(bases_a, bits_a)

        # Bob generates a random set of bases and finds the corresponding
        # angles
        bases_b = self.generate_bases()
        phi_list = np.array([0 if basis == "+" else -45 for basis in bases_b])

        # Eve generates a random set of bases and finds the corresponding
        # angles
        if eve:
            bases_e = self.generate_bases()
            alpha_list = np.array([0 if basis == "+" else -45 for basis in bases_e])
            bits_e = []

        bits_b = []

        # Alice sends each bit to Bob
        for i in range(self.bitlength):
            theta = theta_list[i]
            phi = phi_list[i]
            # Alice sends the photons to Bob
            sent_photons = self.send_photons(theta)

            if eve:
                # Eve selects an angle from her random set of angles
                alpha=alpha_list[i]
                # Eve measures the photons in her chosen basis
                received_photons = self.polarizer(sent_photons, alpha)
                bit_e = self.measurement(received_photons)
                bits_e.append(bit_e)
                beta = self.bit_map[(bases_e[i], bit_e)]
                # Eve sends photons to Bob corresponding to the basis she
                # measured in and the bit she measured
                sent_photons = self.send_photons(beta)

            # Bob measures the photons in his chosen basis
            received_photons = self.polarizer(sent_photons, phi)
            bit_b = self.measurement(received_photons)
            bits_b.append(bit_b)

        bits_b = np.array(bits_b)
        if eve:
            bits_e = np.array(bits_e)
        # Alice and Bob compare their bases and keep the bits where they
        # match
        K_a = np.where(bases_a == bases_b, bits_a, "")
        K_a = K_a[K_a != ""].astype(int)
        K_b = np.where(bases_a == bases_b, bits_b, "")
        K_b = K_b[K_b != ""].astype(int)

        # Alice and Bob compare their keys
        if np.array_equal(K_a, K_b):
            # print("No eavesdropper")
            return bases_a, bits_a, bases_b, bits_b, K_a
        else:
            # print("Eavesdropper detected")
            return bases_a, bits_a, bases_e, bits_e, bases_b, bits_b, K_a, K_b


bb84 = BB84(100, 100)
lengths = []
sims = 1000
for i in range(sims):
    if i % 10 == 0:
        print(i)
    result = bb84.generate_key(eve=False)
    lengths.append(len(result[4]))

def func(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


hist, bin_edges = np.histogram(lengths, bins='auto', density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

popt, pcov = curve_fit(func, bin_centers, hist, p0=[1, np.mean(lengths), np.std(lengths)])

plt.hist(lengths, bins='auto', density=True, alpha=0.5, label='Histogram')
plt.plot(bin_centers, func(bin_centers, *popt), 'r--', label='Fitted Curve')
plt.legend()
plt.xlabel("Length of shared key")
plt.ylabel("Probability density")
plt.grid()
plt.savefig("BB84_histogram2.pdf", bbox_inches='tight', dpi=300)
print(popt, np.sqrt(np.diag(pcov)))
plt.show()
