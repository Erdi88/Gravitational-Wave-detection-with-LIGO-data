import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann, hamming
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


h = h5py.File(DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_h = h["strain/Strain"][()]

L = h5py.File(DATA / "L-L1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_L = L["strain/Strain"][()]

#Sampling rate for both signals
f_s = 4096 #Hz

#Two methods of finding number of samples
print(data_h.shape)
print(data_L.shape)
num_samples_h = len(data_h)
num_samples_L = len(data_L)

H_length_seconds = num_samples_h / f_s 
L_length_seconds = num_samples_L / f_s 

t_H = np.arange(num_samples_h) / f_s
t_L = np.arange(num_samples_L) / f_s

print(H_length_seconds)
print(L_length_seconds)

print(min(data_h))
print(max(data_h))
print(min(data_L))
print(max(data_L))

plt.figure(figsize=(6, 6)) 
plt.subplot(2, 1, 1) 
plt.plot(t_H, data_h, label='H1 signal')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('H1 strain plot')
plt.legend(loc='upper right')
plt.subplot(2, 1, 2) 
plt.plot(t_L, data_L, label='L1 signal', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('L1 strain plot')
plt.legend(loc='upper right')
plt.tight_layout()  
plt.savefig(FIGURES / "Task2.png")
plt.show()


