# Feature list

## Notes

- Dont use first 10 or 20 ms of TEP data (high noise)

- freq bands: delta (0.5-4.0 Hz), theta (4.0-8.0 Hz), alpha (8.0-12.0 Hz), low beta (12.0-18.0 Hz), high beta (18.0-35.0 Hz), low gamma (35.0-58.0 Hz), and high gamma (58.0-100.0 Hz) (haxel 2024)
  - Personal note/hypothesis: delta is associated with deep sleep & relaxation, so maybe not necessary to analyse? Although patients in resting state could generate these waves; gamma is mostly for high alertness so this could be omitted maybe; procedure was TBS so theta band could be most significant

- TEP windows:
  - 25-40, 25-80, 85-150, 160-250 (tautan 2023)
  - 15-35, 45-65, 100-130, 165-195, 275-305 (Bertazzoli 2021)

- statistical features: min, max, mean, std, kurtosis, skew, peak to peak amplitude, RMS

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
  - fd

## Further reading

- phase locking value
- AR coefficients (time domain)
- hilbert transform
