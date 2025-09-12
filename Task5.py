import h5py
import numpy as np
import matplotlib.pyplot as plt

h = h5py.File("H-H1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_h = h["strain/Strain"][()]
L = h5py.File("L-L1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_L = L["strain/Strain"][()]
from scipy.signal.windows import hann

f_s = 4096
N_h = len(data_h)
N_L = len(data_L)

hann_window_h = hann(N_h)
hann_window_L = hann(N_L)
data_h_windowed = hann_window_h * data_h
data_L_windowed = hann_window_L * data_L

fft_h_w = np.fft.fft(data_h_windowed)
fft_L_w = np.fft.fft(data_L_windowed)

magnitude_h = np.abs(fft_h_w)
magnitude_L = np.abs(fft_L_w)

white_filter_h = np.where(magnitude_h != 0, 1 / magnitude_h, 0)
white_filter_L = np.where(magnitude_L != 0, 1 / magnitude_L, 0)

white_dft_h = white_filter_h * fft_h_w
white_dft_L = white_filter_L * fft_L_w

y_h = np.fft.ifft(white_dft_h)
y_L = np.fft.ifft(white_dft_L)

y_h_real = np.real(y_h)
y_L_real = np.real(y_L)

t_h = np.arange(len(y_h_real)) / f_s
t_L = np.arange(len(y_L_real)) / f_s

plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 1)
plt.plot(t_h, y_h_real, color="purple")
plt.title("Whitened Hanford signal (yH[n])")
plt.xlabel("Time (seconds)")
plt.ylabel("Whitened strain y[n]")
plt.xlim(0, 32)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t_L, y_L_real, color="red")
plt.title("Whitened Livingston signal (yL[n])")
plt.xlabel("Time (seconds)")
plt.ylabel("Whitened strain y[n]")
plt.xlim(0, 32)
plt.grid()

plt.tight_layout()
plt.savefig("Task5c.png")
plt.show()
