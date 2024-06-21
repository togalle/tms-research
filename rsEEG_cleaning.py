import os
from ipywidgets import *
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import utils
from autoreject import AutoReject, get_rejection_threshold


logger = utils.get_logger()

# -----------------------------------------------------------------------------

# Cleaning functions

def remove_EOG(eeg_data):
    logger.info("Removing EOG channels")
    eeg_data = eeg_data.drop_channels(["HEOG", "VEOG"])
    return eeg_data


def downsample(eeg_data, sample_rate=1000):
    logger.info(f"Downsampling to {sample_rate} Hz")
    eeg_data = eeg_data.resample(sample_rate, npad="auto")
    return eeg_data


def demean(eeg_data):
    logger.info("Demeaning")
    eeg_data = eeg_data.apply_function(lambda x: x - np.mean(x))
    return eeg_data


def bandpass_notch(eeg_data, l_freq=1, h_freq=100, notch_freqs=[50]):
    logger.info(f"Bandpass filtering between {l_freq} and {h_freq} Hz, and notch filtering at {notch_freqs} Hz")
    eeg_data = eeg_data.filter(l_freq, h_freq)
    eeg_data = eeg_data.notch_filter(notch_freqs)
    return eeg_data


def ICA(eeg_data):
    logger.info("Applying ICA")
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(eeg_data)
    
    # iclabel
    ic_labels = label_components(eeg_data, ica, method="iclabel")
    logger.info(ic_labels)
    labels = ic_labels["labels"]
    proba = ic_labels["y_pred_proba"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"] and proba[idx] > 0.8
    ]
    logger.info(f"icalabel components: {exclude_idx}")
    
    # EOG channels
    eog_indices, eog_scores = ica.find_bads_eog(eeg_data, ch_name="HEOG")
    logger.info(f"EOG components: {eog_indices}")

    # merge indices
    exclude_idx = list(set(exclude_idx + eog_indices))
    logger.info(f"Excluding components: {exclude_idx}")
    
    # remove EOG channels
    eeg_data = ica.apply(eeg_data, exclude=exclude_idx)
    eeg_data = eeg_data.drop_channels(["HEOG", "VEOG"])
    
    return eeg_data


def rereference(eeg_data):
    logger.info("Rereferencing to average")
    eeg_data = eeg_data.set_eeg_reference(ref_channels="average")
    return eeg_data


def rsEEG_epoch(eeg_data, duration=2.0):
    """Create an epochs object from the raw brainvision data, where each epoch is 2 seconds long and got baseline corrected using the last 500ms of the previous epoch

    Args:
        eeg_data (Any): full EEG data
    """
    
    logger.info("Epoching")
    
    events = mne.make_fixed_length_events(eeg_data, duration=duration)
    epochs = mne.Epochs(eeg_data, events, baseline=(0, 0), tmin=0, tmax=duration, preload=True)
    
    return epochs


def autoreject(epochs, transform_bads=True):
    logger.info("Rejecting bad epochs")
    if transform_bads:
        ar = AutoReject(n_jobs=3, picks="eeg")
        epochs_clean = ar.fit_transform(epochs)
    else:
        reject = get_rejection_threshold(epochs, decim=2, ch_types=["eeg"])
        epochs_clean = epochs.drop_bad(reject=reject)
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
    
    # Downsample
    eeg_data = downsample(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Demean
    eeg_data = demean(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Bandpass and notch filter
    eeg_data = bandpass_notch(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # ICA filter
    eeg_data = ICA(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Rereference
    eeg_data = rereference(eeg_data)
    if plot_intermediate:
        utils.plot_rsEEG_raw_average(eeg_data)
    
    # Epoching
    epochs = rsEEG_epoch(eeg_data)
    if plot_intermediate:
        utils.plot_epochs_average(epochs, start=-1, end=1)
    
    # Reject bad epochs
    epochs = autoreject(epochs)
    if plot_intermediate:
        utils.plot_epochs_average(epochs, start=-1, end=1)
    
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
