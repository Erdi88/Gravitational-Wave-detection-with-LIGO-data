import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal.windows import hann, hamming
#parameters
f_s = 4096  
f_1 = 31.5 
f_2 = 1234.56  
n = 4096  

#function to generate the signal x[n]
def x(f_1, f_2, n, f_s):
    return 10e-5 * np.cos((2 * np.pi * f_1 * n) / f_s) + np.cos((2 * np.pi * f_2 * n) / f_s)

#generate sample indices and signal x[n]
N = np.arange(n)
x_n = x(f_1, f_2, N, f_s)

#apply Hann window to the signal
w = hann(len(N))
wx_n = w * x_n

#plot the original and windowed signals
plt.figure(figsize=(8, 6))
plt.plot(N, x_n, label='Original Signal x[n]')
plt.plot(N, wx_n, label='Hann window Signal w[n]x[n]')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Original and Windowed Signal')
plt.legend(loc='upper right')
plt.savefig("Task3b.png")
plt.show()

#compute the FFT of the signals
x_n_fft = np.fft.fft(x_n)
wx_n_fft = np.fft.fft(wx_n)

#convert to decibel scale (power in dB)
power_db_xn = 10 * np.log10(np.abs(x_n_fft)**2)
power_db_wxn = 10 * np.log10(np.abs(wx_n_fft)**2)

#shifted power spectrum 
power_db_xn_shifted = np.fft.fftshift(power_db_xn)
power_db_wxn_shifted = np.fft.fftshift(power_db_wxn)

#frequency axis in Hz
frequency = np.fft.fftfreq(n, d=1/f_s)
frequency_shifted = np.fft.fftshift(frequency)


#Ploting the power spectrum for both original and windowed signals
plt.figure(figsize=(10, 6))
plt.plot(frequency_shifted, power_db_xn_shifted, label='Power spectrum (Original signal)')
plt.plot(frequency_shifted, power_db_wxn_shifted, label='Power spectrum (Hann windowed signal)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Power spectrum for original and windowed signal')
plt.legend()
plt.savefig("Task3d.png")

#Freq 31.5, 1234.56 Hz
for f in [31.5, 1234.56]:
    plt.axvline(x=f, color='red', linestyle=':', label=f'{f} Hz')
    plt.axvline(x=-f, color='green', linestyle=':', label=f'-{f} Hz')
# plt.scatter(x = f_1, color='green', label='31.5 Hz')
# plt.scatter(x = f_2, color='red', label='1234.56 Hz')
plt.legend()
plt.savefig("Task3e.png")
plt.show()


hamming_w = hamming(len(N))
hamming_xn = hamming_w * x_n
hamming_xn_fft = np.fft.fft(hamming_xn)
power_db_hamming = 10*np.log10(np.abs(hamming_xn_fft)**2)
power_db_hamming_shifted = np.fft.fftshift(power_db_hamming)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(frequency_shifted, power_db_hamming_shifted, label='Power Spectrum (Hamming window signal)')
plt.ylabel('Power (dB)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frequency_shifted, power_db_wxn_shifted, label='Power Spectrum (Hann window Signal)', color='orange')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.legend()
plt.grid(True)

plt.suptitle('Power Spectrum for Original and Windowed Signal')
plt.savefig("Task3d2.png")
plt.show()


#Peak points
max_index_original = np.argmax(power_db_xn_shifted)
max_index_hann = np.argmax(power_db_wxn_shifted)

##Frequency, power for peak
peak_xn_freq = frequency_shifted[max_index_original]
peak_xn_power = power_db_xn_shifted[max_index_original]
peak_hann_freq = frequency_shifted[max_index_hann]
peak_hann_power = power_db_wxn_shifted[max_index_hann]

#peak for 32.5
target_freq = 32.5
index_32_5 = np.argmin(np.abs(frequency_shifted - target_freq))
power_32_5_hann = power_db_wxn_shifted[index_32_5]

plt.figure(figsize=(10, 6))
plt.plot(frequency_shifted, power_db_xn_shifted, label="Without Windowing", color='blue')
plt.plot(frequency_shifted, power_db_wxn_shifted, label="Hann Window", color='orange')
for f in [31.5, 1234.56]:
    plt.axvline(x=f, color='green', linestyle=':')#, label=f'31.5 Hz')
    plt.axvline(x=-f, color='green', linestyle=':')#, label=f'1234.56 Hz')

plt.plot(peak_xn_freq, peak_xn_power, 'bo', label=f"Peak (No Window): {peak_xn_freq:.2f} Hz, {peak_xn_power:.2f} dB", color='blue')
plt.plot(-peak_xn_freq, peak_xn_power, 'bo', color='blue')
plt.plot(peak_hann_freq, peak_hann_power, 'go', label=f"Peak (Hann): {peak_hann_freq:.2f} Hz, {peak_hann_power:.2f} dB", color='orange')
plt.plot(-peak_hann_freq, peak_hann_power, 'go', color='orange')
plt.plot(target_freq, power_32_5_hann, 'gs', label=f"32.5 Hz (Hann): {power_32_5_hann:.2f} dB", color='orange')
plt.plot(-target_freq, power_32_5_hann, 'gs', color='orange')

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.title("Power Spectrum Comparison: Rectangular and Hann Windows")
# plt.legend(loc='lower left')
plt.legend()
plt.grid()
plt.savefig('Task3e2.png')
plt.show()