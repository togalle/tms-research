import os
from ipywidgets import *
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import utils
import logging
import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s[%(asctime)s] - %(levelname)s - %(message)s'))

logger = colorlog.getLogger("rsEEG_cleaning")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------

# Plotting utilities

def plot_single_response(eeg_data, channel="Pz", tmin=-0.005, tmax=0.01):
    events, event_dict = mne.events_from_annotations(eeg_data)
    event_id = event_dict["Stimulus/S  1"]
    epochs = mne.Epochs(
        eeg_data,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        picks=channel,
    )

    epochs.plot(picks=channel, n_epochs=1, show=True, scalings={"eeg": 50e-4})


def plot_average_epoch(epochs, start=-0.05, end=0.25, electrodes=None):
    epochs = epochs.copy()
    if electrodes is not None:
        epochs.pick_channels(electrodes)
    data = epochs.get_data(copy=False)
    mean_responses = np.mean(data, axis=0)
    time_points = np.linspace(-1, 1, data.shape[2])
    selected_indices = np.where((time_points >= start) & (time_points <= end))
    for i, mean_response in enumerate(mean_responses):
        selected_data = mean_response[selected_indices]
        selected_time_points = time_points[selected_indices]
        plt.plot(selected_time_points, selected_data, label=f"Channel {i+1}")
    plt.xlabel("Time points")
    plt.ylabel("Mean response")
    plt.show()


def plot_response(eeg):
    utils.plot_average_response(eeg, tmin=-0.05, tmax=0.25)  # Check full response
    utils.plot_single_response(
        eeg, channel="Pz", tmin=-0.05, tmax=0.05
    )  # Check for TMS pulse

def plot_full_average_epoch(epochs, electrodes=None, start=-0.05, end=0.25, plot_gmfp=True):
    epochs = epochs.copy()
    if electrodes is not None:
        epochs.pick_channels(electrodes)
    data = epochs.get_data(copy=False)
    mean_responses = np.mean(data, axis=(0, 1))
    sem_responses = np.std(data, axis=(0, 1)) / np.sqrt(data.shape[0])
    time_points = np.linspace(-1, 1, data.shape[2])
    selected_indices = np.where((time_points >= start) & (time_points <= end))
    selected_data = mean_responses[selected_indices]
    selected_sem = sem_responses[selected_indices]
    selected_time_points = time_points[selected_indices]
    plt.plot(selected_time_points, selected_data, label="Average of all electrodes")
    plt.fill_between(
        selected_time_points,
        selected_data - selected_sem,
        selected_data + selected_sem,
        color="b",
        alpha=0.2,
    )
    if plot_gmfp:
        gmfp = np.std(data, axis=1).mean(axis=0)[selected_indices]
        plt.plot(selected_time_points, gmfp, label="GMFP")
    plt.xlabel("Time points")
    plt.ylabel("Mean response")
    plt.legend()
    plt.show()

# -----------------------------------------------------------------------------

# Cleaning functions

def remove_EOG(eeg_data):
    logger.info("Removing EOG channels")
    eeg_data.drop_channels(["HEOG", "VEOG"])


def downsample(eeg_data, sample_rate=1000):
    logger.info(f"Downsampling to {sample_rate} Hz")
    eeg_data.resample(sample_rate, npad="auto")


def demean(eeg_data_raw):
    eeg_data_raw.apply_function(lambda x: x - np.mean(x))
    

def bandpass_notch(eeg_data, l_freq=1, h_freq=100, notch_freqs=[50]):
    eeg_data.filter(l_freq, h_freq)
    eeg_data.notch_filter(notch_freqs)
    


def ICA(eeg_data):
    ica = ICA(n_components=20, random_state=97)
    ica.fit(eeg_data)
    ic_labels = label_components(eeg_data, ica, method="iclabel")
    
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    
    print(f"Excluding ICA components: {exclude_idx}") 
    
    ica.apply(eeg_data, exclude=exclude_idx)


def rereference(eeg_data):
    eeg_data.set_eeg_reference(ref_channels="average")

# -----------------------------------------------------------------------------

# Pipeline function

def clean_spTEP(
    filename,
    eeg_data_raw,
    plot_intermediate=False,
    plot_result=False,
    finalplot_electrodes=None,
    save_result=True,
):
    mne.set_log_level("WARNING")

    eeg_data = eeg_data_raw.copy()

    # Remove EOG
    remove_EOG(eeg_data)
    if plot_intermediate:
        plot_response(eeg_data)
    
    # Downsample
    downsample(eeg_data)
    if plot_intermediate:
        plot_response(eeg_data)
    
    # Demean
    demean(eeg_data)
    if plot_intermediate:
        plot_response(eeg_data)
    
    # Bandpass and notch filter
    bandpass_notch(eeg_data)
    if plot_intermediate:
        plot_response(eeg_data)
    
    # ICA filter
    ICA(eeg_data)
    if plot_intermediate:
        plot_response(eeg_data)
    
    # Rereference
    rereference(eeg_data)
    if plot_intermediate:
        plot_response(eeg_data)
    
    # TODO: epoching
    
    if save_result:
        filename = os.path.basename(filename)
        filename_base, filename_ext = os.path.splitext(filename)
        filename = filename_base + "_filtered-epo.fif"
        foldername = "filtered"
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        eeg_data.save(os.path.join(foldername, filename), overwrite=True)
