import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann


h = h5py.File("H-H1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_h = h["strain/Strain"][()]  

L = h5py.File("L-L1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_L = L["strain/Strain"][()]  

f_s = 4096  
N_h = len(data_h)  
N_L = len(data_L) 


hann_window_h = hann(N_h)
hann_window_L = hann(N_L)
data_h_windowed = hann_window_h * data_h
data_L_windowed = hann_window_L * data_L


fft_h = np.fft.fft(data_h_windowed)
fft_L = np.fft.fft(data_L_windowed)


power_db_h = 10 * np.log10(np.abs(fft_h)**2)
power_db_L = 10 * np.log10(np.abs(fft_L)**2)


freq_h = np.fft.fftfreq(N_h, d=1/f_s)
freq_L = np.fft.fftfreq(N_L, d=1/f_s)


positive_freq_h = freq_h[freq_h >= 0]
positive_power_db_h = power_db_h[freq_h >= 0]

positive_freq_L = freq_L[freq_L >= 0]
positive_power_db_L = power_db_L[freq_L >= 0]

strong_freq_h = [5.4, 59.7, 121.7, 178.6, 332.2, 500.1, 996.1, 1084.4,1456.2, 1485.5, 1940.1]
strong_power_h = [-288.6, -333.6, -355.8, -356.7, -337.7, -331.3, -310.2, -344.1, -322.0, -320.0, -360.6]

strong_freq_L= [-1.4, 179.4, 330.5, 515.6, 1012.2, 1083.5, 1273.7, 1497.8, 1665.0, 1963.0]
strong_power_L = [-263.1, -343.4, -335.9, -299.2, -311.2, -346.8, -355.4, -329.8, -351.1, -376.4] 

plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 1)
plt.plot(positive_freq_h, positive_power_db_h, label='Power spectrum (Hanford)', color='blue')
# plt.scatter(strong_freq_h, strong_power_h, color='purple', label="Strong spectral components")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Power Spectrum for Hanford Data')
plt.xlim(-50, f_s / 2 + 50)
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(positive_freq_L, positive_power_db_L, label='Power spectrum (Livingston)', color='orange')
# plt.scatter(strong_freq_L, strong_power_L, color='red', label="Strong spectral components")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Power Spectrum for Livingston Data')
plt.xlim(-50, f_s / 2 + 50)
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("Task4b.png")
plt.show()

