# Feature list

These notes were made during the creation of the feature extraction scripts, to better understand which ones are commonly used and important details to understand about them.

## Notes

- Dont use first 10 or 20 ms of TEP data (high noise)

- freq bands: delta (0.5-4.0 Hz), theta (4.0-8.0 Hz), alpha (8.0-12.0 Hz), low beta (12.0-18.0 Hz), high beta (18.0-35.0 Hz), low gamma (35.0-58.0 Hz), and high gamma (58.0-100.0 Hz) (haxel 2024)
  - Personal note/hypothesis: delta is associated with deep sleep & relaxation, so maybe not necessary to analyse? Although patients in resting state could generate these waves; gamma is mostly for high alertness so this could be omitted maybe; procedure was TBS so theta band could be most significant

- TEP windows:
  - 25-40, 25-80, 85-150, 160-250 (tautan 2023)
  - 15-35, 45-65, 100-130, 165-195, 275-305 (Bertazzoli 2021)

- statistical features: min, max, mean, std, kurtosis, skew, peak to peak amplitude, RMS

rsEEG:
- time-domain
  - Statistical features
    - Mean
    - STD
- PCA, ICA
- Connectivity
- Hjorth parameters

spTEP:
- latency of peaks
- GMFA
- Connectivity
- frequency-domain
  - FFT (Welch method)
  - PSD
- time-frequency-domain:
  - Wavelet transform
  - spectrogram

TEPs:
  - ROI (single electrode or average) or GMFA (TESA)
  - User defines latency and time window (TESA)
  - AUC (TESA)
  - latency & amplitude  of TEP & GMFA (Farzan 2016)

---

Zandvakili 2021
- rsEEG into 2s epochs, only > 120 s of usable data (sixty 2-s epochs) was used
- The **power spectral density** of artifact-free 2-s epochs was calculated using a **Welch Power Spectral Density estimate and a Hamming window with a 50% overlap** (MATLAB Signal Processing Toolbox).
- **Power** was calculated for frequency bands corresponding to delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz)
- **Coherence** was calculated based on spectral density

farzan 2016
- connectivity is used to measure connection between two regions to form a bigger functional unit
  - undirected connectivity (no quantification of direction of info flow): correlation, coherence, synchrony
  - with direction (computationally expensive): directed transfer function, partial directed coherence
- TEP, frequency-domain power (PSD), time-frequency, phase-domain analysis
- negativity at 15 ms (N15), a positivity at 30 ms (P30), followed by N45, P55, N100, P180, and N280
- GMFA
- any eeg feat.extr. for eeg can be used on tms

haxel 2024
- delta (0.5-4.0 Hz), theta (4.0-8.0 Hz), alpha (8.0-12.0 Hz), low beta (12.0-18.0 Hz), high beta (18.0-35.0 Hz), low gamma (35.0-58.0 Hz), and high gamma (58.0-100.0 Hz)
- eeg power asymmetry

Gil Avila 2023
- functional connectivity!! phase- or amplitude based
- power spectrum (averaged across channels & trials), alpha peak frequency (frequency in alpha band with greatest value in PSD or center of gravity of power spectra in alpha band), projection to source space to compute functional connectivity on ROI's in source space

shoorangiz (eeg) 2021
- time domain
  - zero crossing
  - hjorth parameters
    - activity (variance)
    - mobility (mean frequency)
    - complexity (bandwith of signal)
  - nonlinear energy
- freq domain (using FFT)
  - PSD using Welch's method
  - Spectral entropy
  - intensity-weighted mean frequency (weighted average vreq of signal relative to its PSD)
  - intensity-weighted bandwidth (variance of frequency)

li 2018
- time & frequency domain
  - peak-peak mean (time)
  - mean square (time)
  - variance (time)
  - hjorth parameters (freq)
  - maximum PS freq (freq)
  - max PS density (freq)
  - power sum (freq)
- Non-linear Dynamical System Features (mostly just a bunch of different types of entropy)

HTNet gebruikt geen features, gewoon raw EEG data

Glaser 2020 mentions that modern algorithms (neural network & gradient boost) significantly outperform traditional ones (wiener/kalman filter) (for neural decoding)

**Gemein** 2020
- FFT
  - max, mean, min, peak freq, power, power ratio, spectral entropy, value range, variance
- patient age
- phase locking value (connectivity)
- Time (using PyEEG)
  - energy
  - hjorth parameters
  - kurtosis
  - line length
  - mean, max, min, median, skewness
  - zero crossings
  - nonlinear energy

saeidi 2021
- mean, standard deviation, variance, root mean square, skewness, kurtosis, relative band energy, and entropy
- FFT with Welch
- PCA
- AR
- FFT
- wavelet & STFT

tautan 2023
- windows: 25-40, 25-80, 85-150, 160-250
- first normalized between -1 and 1, then max, min, mean, skew, kurtosis
- hjorth parameters
- total signal energy 
- TEP peaks (peak amplitude is average around -5 and 5ms before 80ms and -15 and 15ms after)
- Mean field power & AUC 

**singh and krishnan** 2022
- time-domain
  - AR
  - fractal dimension (index that measures signal complexity through mathematical means)
  - statistical features (mean, std)
  - detrended fluctuation analyis (observe the presence or absence of long-range temporal correlations (LRTC))
- frequency-domain
  - fourier transform (FFT preferrably)
  - PSD
  - band power
  - Hilbert transform

zandvakili 2019
- psd & power in bands
- coherence

## Features

- Patient information (e.g. age)

- Time domain
  - on GMFA: statistical features
  - correlation, coherence
  - zero crossings
  - line length
  - Hjorth
  - TEP peak latency & amplitude (in each window; either in GMFA or in source electrodes)

- Frequency domain (Psd using Welch's method)
  - Power of frequency bands
  - Correlation, coherence
  - Statistical features
  - Power asymmetry (for each pair of opposing electrodes)
  - Alpha peak frequency (correlated with behavioral and cognitive characteristics43 in aging and disease)
  - Spectral entropy (Shoorangiz 2021, Gemein 2020)
  - Intensity-weighted bandwidth (variance of frequency) (Shoorangiz 2021)
  - Intensity-weighted mean frequency (weighted average freq of signal relative to its PSD) (Shoorangiz 2021)
  - Max PS freq & density
  - Hjorth
  - AUC

- Time-frequency domain (Wavelet, STFT, HHT)
