import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann, hamming

# Parameters
f_s = 4096  # Sampling rate in Hz
f_1 = 31.5  # Frequency of the weak sinusoid
f_2 = 1234.56  # Frequency of the strong sinusoid
n = 4096  # Number of samples

# Function to generate the signal x[n]
def x(f_1, f_2, n, f_s):
    return 10e-5 * np.cos((2 * np.pi * f_1 * n) / f_s) + np.cos((2 * np.pi * f_2 * n) / f_s)

# Generate sample indices and signal x[n]
N = np.arange(n)
x_n = x(f_1, f_2, N, f_s)

# Apply Hann and Hamming windows
hann_window = hann(len(N))
hamming_window = hamming(len(N))
hann_x_n = hann_window * x_n
hamming_x_n = hamming_window * x_n

# Compute the FFT of the signals
hann_x_n_fft = np.fft.fft(hann_x_n)
hamming_x_n_fft = np.fft.fft(hamming_x_n)

# Convert to decibel scale (power in dB)
power_db_hann = 10 * np.log10(np.abs(hann_x_n_fft)**2)
power_db_hamming = 10 * np.log10(np.abs(hamming_x_n_fft)**2)

# Shifted power spectrum for display (centering zero frequency)
power_db_hann_shifted = np.fft.fftshift(power_db_hann)
power_db_hamming_shifted = np.fft.fftshift(power_db_hamming)

# Generate frequency axis in Hz, then shift for plotting
frequency = np.fft.fftfreq(n, d=1/f_s)
frequency_shifted = np.fft.fftshift(frequency)

# Plot the power spectrum for both Hann and Hamming windowed signals
plt.figure(figsize=(10, 6))
plt.plot(frequency_shifted, power_db_hann_shifted, label='Hann Window Power Spectrum')
plt.plot(frequency_shifted, power_db_hamming_shifted, label='Hamming Window Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Power Spectrum of Hann and Hamming Windowed Signals')
plt.legend()

# Mark the frequencies of interest: 31.5 Hz and 1234.56 Hz
plt.axvline(x=f_1, color='red', linestyle='--', label='f1 = 31.5 Hz')
plt.axvline(x=f_2, color='green', linestyle='--', label='f2 = 1234.56 Hz')
plt.legend()
plt.show()
