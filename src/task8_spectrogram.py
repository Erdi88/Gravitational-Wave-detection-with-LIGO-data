import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"



h = h5py.File(DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_h = h["strain/Strain"][()]  

L = h5py.File(DATA / "L-L1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_L = L["strain/Strain"][()]  

f_s = 4096  #Hz

hann_window_h = hann(len(data_h))
hann_window_L = hann(len(data_L))
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

L = 7
running_mean_filter = np.ones(L) / L

y_h_filtered = np.convolve(y_h_real, running_mean_filter, mode='same')
y_L_filtered = np.convolve(y_L_real, running_mean_filter, mode='same')


nperseg = int(f_s*0.1)  
noverlap = f_s*0.09

frequencies_h, times_h, Sxx_h = spectrogram(y_h_filtered, fs=f_s, window='hann', nperseg=nperseg, noverlap=noverlap)

frequencies_L, times_L, Sxx_L = spectrogram(y_L_filtered, fs=f_s, window='hann', nperseg=nperseg, noverlap=noverlap)


Sxx_h_dB = 10 * np.log10(np.abs(Sxx_h)**2)
Sxx_L_dB = 10 * np.log10(np.abs(Sxx_L)**2)


plt.figure(figsize=(8, 6))
plt.pcolormesh(times_h, frequencies_h, Sxx_h_dB, vmin=-190)
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.title('Dynamic power spectrum of Hanford signal')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.xlim(15.5, 17) 
plt.ylim(0, 400)  
plt.tight_layout()
plt.savefig(FIGURES / "Task8bh.png")
plt.show()


plt.figure(figsize=(8, 6))
plt.pcolormesh(times_L, frequencies_L, Sxx_L_dB, vmin=-190)
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.title('Dynamic power spectrum of Livingston signal')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.xlim(15.5, 17) 
plt.ylim(0, 400)  
plt.tight_layout()
plt.savefig(FIGURES / "Task8bL.png")
plt.show()

