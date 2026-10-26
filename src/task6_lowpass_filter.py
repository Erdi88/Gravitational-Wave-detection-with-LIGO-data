import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


#LIGO data from files
h = h5py.File(DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_h = h["strain/Strain"][()]  

L = h5py.File(DATA / "L-L1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_L = L["strain/Strain"][()] 

f_s = 4096  # Hz
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

#Whitening filter
white_filter_h = np.where(magnitude_h != 0, 1 / magnitude_h, 0)
white_filter_L = np.where(magnitude_L != 0, 1 / magnitude_L, 0)

white_dft_h = white_filter_h * fft_h_w
white_dft_L = white_filter_L * fft_L_w

#Inverse FFT
y_h = np.fft.ifft(white_dft_h)
y_L = np.fft.ifft(white_dft_L)

#REal part
y_h_real = np.real(y_h)
y_L_real = np.real(y_L)


t_h = np.arange(len(y_h_real)) / f_s 
t_L = np.arange(len(y_L_real)) / f_s  


f_cutoff = 300 
# L_filter = (f_s / (2 * f_cutoff)) 
# print(f"Filter length L: {L_filter}")

L_filter = 8

running_mean_filter = np.ones(L_filter) / L_filter


freqs = np.fft.fftfreq(4096, d=1/f_s)#[:2048]  
H = np.fft.fft(running_mean_filter, n=4096)#[:2048]  
power_response_dB = 10 * np.log10(np.abs(H)**2)

plt.figure(figsize=(10, 5))
plt.plot(freqs, power_response_dB, label="Power Spectral Response")
plt.axhline(-6, color='red', linestyle='--', label="-6 dB point")
plt.axvline(f_cutoff, color='green', linestyle='--', label="300 Hz cutoff")
plt.title("Power spectral response on running low pass filter (L=8)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.legend()
plt.grid()
plt.savefig(FIGURES / "Task6b.png")
plt.show()

#time dely
tau = (L_filter - 1) / 2 / f_s
print(f"Time delay τ: {tau:.6f} seconds")


y_h_filtered = np.convolve(y_h_real, running_mean_filter, mode='same')
y_L_filtered = np.convolve(y_L_real, running_mean_filter, mode='same')

t_h_filtered = np.arange(len(y_h_filtered)) / f_s
t_L_filtered = np.arange(len(y_L_filtered)) / f_s

t_h_shifted = t_h_filtered - tau
t_L_shifted = t_L_filtered - tau

plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 1)
plt.plot(t_h_shifted, y_h_filtered, color="orange")
plt.title("Low-Pass filter: Whitened Hanford signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Filtered whitened strain")
plt.xlim(16.1, 16.6)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t_L_shifted, y_L_filtered, color="blue")
plt.title("Low-Pass filter: Whitened Livingston signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Filtered whitened strain")
plt.xlim(16.1, 16.6)
plt.grid()

plt.tight_layout()
plt.savefig(FIGURES / "Task6f-a.png")
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(t_h_shifted, y_h_filtered, color="blue", label="Whitened Hanford Signal")
plt.plot(t_L_shifted, y_L_filtered, color="orange", label="Whitened Livingstone signal")
plt.title("Low-Pass filter: Whitened Hanford and Livingstone Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Filtered whitened strain")
plt.xlim(16.1, 16.6)  
plt.grid()
plt.legend()
plt.savefig(FIGURES / "Task6f-b.png")
plt.show()

