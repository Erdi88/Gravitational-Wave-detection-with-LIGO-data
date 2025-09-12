# Gravitational Wave Detection with LIGO Data

## Overview

This repository contains my solution for **Programming Assignment 3: Gravitational Wave Detection with LIGO Data**. The goal of this assignment is to apply signal processing techniques to detect gravitational wave signals from the LIGO detectors (Hanford and Livingston).

The assignment focuses on analyzing real LIGO data to verify the detection of gravitational waves using:

- Discrete Fourier Transforms (DFT)  
- Tapered windowing (Hann and Hamming)  
- Whitening filters  
- Low-pass filters  
- Time-frequency analysis (dynamic spectrum / spectrogram)  

> **Note:** This project is educational and focuses on signal processing rather than the physics of LIGO.

## Features

- Reading HDF5 LIGO data for Hanford (H1) and Livingston (L1) detectors.  
- Plotting raw strain signals in the time domain.  
- Applying tapered window functions for improved spectral analysis.  
- Computing and plotting power spectra to identify narrowband interference.  
- Implementing whitening filters in the frequency domain.  
- Applying low-pass filtering to highlight the gravitational wave signal.  
- Estimating time delays between the two detectors.  
- Calculating and visualizing dynamic spectra (spectrograms).  

## Requirements

- Python 3.7+  
- `numpy`  
- `scipy`  
- `matplotlib`  
- `h5py`  

> Tested on Python 3.10.

## Usage

1. **Download LIGO data files** from the course repository:  
   - `H-H1_LOSC_4_V2-1126259446-32.hdf5`  
   - `L-L1_LOSC_4_V2-1126259446-32.hdf5`  

2. Place the files in the same directory as the Python scripts.

3. Run the main analysis script:

```bash
python ligo_analysis.py
