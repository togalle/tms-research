import os
from ipywidgets import *
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import utils
from autoreject import AutoReject


logger = utils.get_logger()

# -----------------------------------------------------------------------------

# Cleaning functions

def remove_EOG(eeg_data):
    logger.info("Removing EOG channels")
    eeg_data.drop_channels(["HEOG", "VEOG"])


def downsample(eeg_data, sample_rate=1000):
    logger.info(f"Downsampling to {sample_rate} Hz")
    eeg_data.resample(sample_rate, npad="auto")


def demean(eeg_data_raw):
    logger.info("Demeaning")
    eeg_data_raw.apply_function(lambda x: x - np.mean(x))


def bandpass_notch(eeg_data, l_freq=1, h_freq=100, notch_freqs=[50]):
    logger.info(f"Bandpass filtering between {l_freq} and {h_freq} Hz, and notch filtering at {notch_freqs} Hz")
    eeg_data.filter(l_freq, h_freq)
    eeg_data.notch_filter(notch_freqs)

def ICA(eeg_data):
    logger.info("Applying ICA")
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(eeg_data)
    ic_labels = label_components(eeg_data, ica, method="iclabel")
    
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    
    print(f"Excluding ICA components: {exclude_idx}") 
    
    ica.apply(eeg_data, exclude=exclude_idx)


def rereference(eeg_data):
    logger.info("Rereferencing to average")
    eeg_data.set_eeg_reference(ref_channels="average")


def rsEEG_epoch(eeg_data, duration=2.0):
    """Create an epochs object from the raw brainvision data, where each epoch is 2 seconds long and got baseline corrected using the last 500ms of the previous epoch

    Args:
        eeg_data (Any): full EEG data
    """
    
    logger.info("Rereferencing to average")
    
    events = mne.make_fixed_length_events(eeg_data, duration=duration)
    epochs = mne.Epochs(eeg_data, events, baseline=(None, None), tmin=0, tmax=duration, preload=True)
    
    return epochs


def autoreject(epochs):
    logger.info("Rejecting bad epochs")
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    return epochs_clean

# -----------------------------------------------------------------------------

# Pipeline function

def clean_rsEEG(
    eeg_data_raw,
    plot_intermediate=False,
    save_result=False,
    filename=None,
):
    mne.set_log_level("WARNING")

    eeg_data = eeg_data_raw.copy()

    # Remove EOG
    remove_EOG(eeg_data)
    if plot_intermediate:
        # plot_response(eeg_data)
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Downsample
    downsample(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Demean
    demean(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Bandpass and notch filter
    bandpass_notch(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # ICA filter
    ICA(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Rereference
    rereference(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Epoching
    epochs = rsEEG_epoch(eeg_data)
    if plot_intermediate:
        utils.plot_epochs_average(epochs)
    
    # Reject bad epochs
    epochs = autoreject(epochs)
    if plot_intermediate:
        utils.plot_epochs_average(epochs)
    
    if save_result:
        if filename == None:
            filename = eeg_data.filename
        filename = os.path.basename(filename)
        filename_base, filename_ext = os.path.splitext(filename)
        filename = filename_base + "_filtered-epo.fif"
        foldername = "filtered"
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        eeg_data.save(os.path.join(foldername, filename), overwrite=True)
    else:
        return epochs
