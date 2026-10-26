import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.signal import correlate
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


h = h5py.File(DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_h = h["strain/Strain"][()]  

L = h5py.File(DATA / "L-L1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_L = L["strain/Strain"][()]  


f_s = 4096  # Hz


hann_window_h = hann(len(data_h))
hann_window_L = hann(len(data_L))
data_h_windowed = hann_window_h * data_h
data_L_windowed = hann_window_L * data_L

#fft windowed signal
fft_h_w = np.fft.fft(data_h_windowed)
fft_L_w = np.fft.fft(data_L_windowed)

#Calculate magnitude
magnitude_h = np.abs(fft_h_w)
magnitude_L = np.abs(fft_L_w)

#construct whitening filter
white_filter_h = np.where(magnitude_h != 0, 1 / magnitude_h, 0)
white_filter_L = np.where(magnitude_L != 0, 1 / magnitude_L, 0)

#apply filter
white_dft_h = white_filter_h * fft_h_w
white_dft_L = white_filter_L * fft_L_w

#Inverse fft
y_h = np.fft.ifft(white_dft_h)
y_L = np.fft.ifft(white_dft_L)

#reaal part and magnitude of the whitened signals
y_h_real = np.real(y_h)
y_L_real = np.real(y_L)


L = 8
running_mean_filter = np.ones(L) / L

y_h_filtered = np.convolve(y_h_real, running_mean_filter, mode='same')
y_L_filtered = np.convolve(y_L_real, running_mean_filter, mode='same')

y_h_mag = np.abs(y_h_filtered)
y_L_mag = np.abs(y_L_filtered)

N = np.arange(len(y_h_mag))

# correlation = correlate(y_L_mag, y_h_mag, mode='full')
# n0 = np.argmax(correlation) - (len(y_h_mag) - 1)  # Index shift to get the delay
# print(f'Optimal sample delay n0: {n0}')

n_0 = np.argmax(y_h_mag) - np.argmax(y_L_mag)
print(n_0)

n_0 = 19
tau = n_0/f_s
print(f'The corresponding tau to the sample delay is {tau}')    

plt.figure(figsize=(10, 8))
plt.plot(N + 19, y_h_mag, label="$|y_{h}[n + n_0]|$", color="red")
plt.plot(N, y_L_mag, label="$|y_{L}[n]|$", color="blue")
plt.title("Time delay")
plt.xlabel("time")
plt.ylabel("Magnitude")
plt.xlim(16.3*f_s, 16.5*f_s)
plt.legend()
plt.savefig(FIGURES / "Task7a1.png")
plt.show()

