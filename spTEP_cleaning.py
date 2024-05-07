# Configuration

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

logger = colorlog.getLogger("spTEP_cleaning")
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


def calculate_range_indices(tms_index, start, end, sampling_rate):
    """
    start and end are positive in seconds
    sampling rate in Hz
    """
    samples_before = int(start * sampling_rate)
    samples_after = int(end * sampling_rate)

    start_index = max(0, tms_index - samples_before)
    end_index = tms_index + samples_after

    return start_index, end_index


def interpolate_TMS_pulse(eeg_data_raw, tms_indices, start, end, sampling_rate):
    logger.info(f"Interpolating TMS pulse from {start} to {end} seconds")
    eeg_data = eeg_data_raw.get_data()
    num_electrodes = eeg_data.shape[0]
    for tms_index in tms_indices:
        start_index, end_index = calculate_range_indices(
            tms_index, start, end, sampling_rate
        )
        for i in range(num_electrodes):
            x = [start_index - 2, start_index - 1, end_index + 1, end_index + 2]
            y = [
                eeg_data[i, start_index - 2],
                eeg_data[i, start_index - 1],
                eeg_data[i, end_index + 1],
                eeg_data[i, end_index + 2],
            ]
            x_new = np.arange(start_index, end_index + 1)

            interp_func = interp1d(x, y, kind="cubic")
            eeg_data[i, start_index : end_index + 1] = interp_func(x_new)

    eeg_data_raw._data = eeg_data


def downsample(eeg_data, sample_rate=1000):
    logger.info(f"Downsampling to {sample_rate} Hz")
    eeg_data.resample(sample_rate, npad="auto")


def epoching(eeg_data):
    logger.info("Epoching")
    events, event_dict = mne.events_from_annotations(eeg_data)
    event_id = event_dict["Stimulus/S  1"]
    epochs = mne.Epochs(
        eeg_data,
        events,
        event_id=event_id,
        tmin=-1,
        tmax=1,
        baseline=None,
        preload=True,
    )
    return epochs


def demean_epochs(epochs):
    logger.info("Demeaning")
    data = epochs.get_data(copy=False)
    demeaned_data = data - np.mean(data, axis=2, keepdims=True)
    demeaned_epochs = mne.EpochsArray(
        demeaned_data, epochs.info, events=epochs.events, event_id=epochs.event_id
    )
    return demeaned_epochs


def ICA_1(epoch_data, T=3.5, b1=0.011, b2=0.030, n_components=20):
    logger.info(f"ICA 1 from {b1} to {b2} seconds")
    ica = ICA(n_components=n_components, random_state=97)
    ica.fit(epoch_data)

    # Credits to Arne Callaert for the following code
    sources = ica.get_sources(epoch_data)
    averaged_sources = sources.get_data().mean(axis=0)
    times = sources.times
    sfreq = sources.info["sfreq"]
    indices = np.where((times >= (b1 / 1000)) & (times <= (b2 / 1000)))
    # print("indices:", indices)
    components_to_remove = []

    for i, component in enumerate(averaged_sources):
        base = len(times) / 2
        b1_index = int(base + (b1 * sfreq))
        b2_index = int(base + (b2 * sfreq))
        x = np.mean(np.abs(component[b1_index:b2_index]))
        y = np.mean(np.abs(component))
        if x / y > T:
            # print("FOUND:", x / y)
            components_to_remove.append(i)
    logger.info(f"Excluding ICA components {components_to_remove}")
    ica.exclude = components_to_remove

    epoch_data = ica.apply(epoch_data)


def bandpass_notch(epoch_data, low_freq=1, high_freq=100, notch_freqs=[50]):
    logger.info(f"Bandpass {low_freq}-{high_freq} Hz and notch {notch_freqs} Hz")
    
    # Bandpass
    epoch_data.filter(low_freq, high_freq)

    # Notch (only directly available on raw object, not on epochs)
    data = epoch_data.get_data(copy=False)
    notch_filtered = mne.filter.notch_filter(
        data, epoch_data.info["sfreq"], notch_freqs
    )
    filtered_epochs = mne.epochs.EpochsArray(
        notch_filtered, epoch_data.info, events=epoch_data.events, tmin=epoch_data.tmin
    )
    return filtered_epochs


def rereference(epochs, ref_channels="average"):
    logger.info("Rereferencing to average")
    epochs.set_eeg_reference(ref_channels)


def ICA_2(epoch_data):
    logger.info("ICA 2")
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(epoch_data)
    ic_labels = label_components(epoch_data, ica, method="iclabel")

    # print(ic_labels["labels"])

    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    # print(f"Excluding these {len(exclude_idx)} ICA components: {exclude_idx}")

    ica.apply(epoch_data, exclude=exclude_idx)


def baseline(epoch_data, period=(-0.500, -0.005)):
    logger.info(f"Baseline correction from {period[0]} to {period[1]} s")
    epoch_data.apply_baseline((1 + period[0], 1 + period[1]))   # Epoch is centered on 1


def remove_epoch_portion(epochs, start_time=-0.005, end_time=0.015):
    # Get the times from the epochs
    times = epochs.times

    # Find the indices of the start and end times
    start_idx = (np.abs(times - start_time)).argmin()
    end_idx = (np.abs(times - end_time)).argmin()

    # Remove the data within the time range from each epoch
    epochs._data = np.delete(epochs._data, np.s_[start_idx:end_idx], axis=-1)

    # Remove the times within the time range
    epochs.times = np.delete(epochs.times, np.s_[start_idx:end_idx])

    return epochs


# -----------------------------------------------------------------------------

def clean_spTEP(
    filename,
    eeg_data_raw,
    plot_intermediate=False,
    interpolate_start=0.005,
    interpolate_end=0.015,
    ICA1_T=3.5,
    ICA1_b1=0.011,
    ICA1_b2=0.030,
    ICA1_n_components=20,
    plot_result=False,
    finalplot_electrodes=None,
    finalplot_start=-0.05,
    finalplot_end=0.25,
    save_result=True,
):
    mne.set_log_level("WARNING")

    eeg_data_copy = eeg_data_raw.copy()
    events, event_dict = mne.events_from_annotations(eeg_data_raw)
    tms_indices = [event[0] for event in events if event[2] == 1]

    # Remove EOG channels
    remove_EOG(eeg_data_copy)
    
    # Remove TMS pulse and interpolate
    interpolate_TMS_pulse(
        eeg_data_copy,
        tms_indices,
        interpolate_start,
        interpolate_end,
        eeg_data_copy.info["sfreq"],
    )
    if plot_intermediate:
        plot_response(eeg_data_copy)
    
    # Downsample
    downsample(eeg_data_copy)
    
    # Epoch
    epochs = epoching(eeg_data_copy)
    if plot_intermediate:
        plot_average_epoch(epochs)
    
    # Demeaning
    epochs = demean_epochs(epochs)
    if plot_intermediate:
        plot_average_epoch(epochs)
    
    # First ICA filter
    ICA_1(
        epochs,
        T=ICA1_T,
        b1=ICA1_b1,
        b2=ICA1_b2,
        n_components=ICA1_n_components,
    )
    if plot_intermediate:
        plot_average_epoch(epochs)
    
    # Bandpass and notch filter
    epochs = bandpass_notch(epochs)
    if plot_intermediate:
        plot_average_epoch(epochs)
    
    # Second ICA filter
    ICA_2(epochs)
    if plot_intermediate:
        plot_average_epoch(epochs)
    
    # Rereference (average)
    rereference(epochs)
    if plot_intermediate:
        plot_average_epoch(epochs)
    
    # Baseline correction
    baseline(epochs)

    if plot_result:
        
        plot_average_epoch(epochs)
        plot_average_epoch(epochs, electrodes=["FC1"])
        plot_full_average_epoch(
            epochs, finalplot_electrodes, finalplot_start, finalplot_end
        )
        plot_full_average_epoch(
            epochs, None, finalplot_start, finalplot_end
        )
        
    if save_result:
        filename = os.path.basename(filename)
        filename_base, filename_ext = os.path.splitext(filename)
        filename = filename_base + "_filtered-epo.fif"
        foldername = "filtered"
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        epochs.save(os.path.join(foldername, filename), overwrite=True)
