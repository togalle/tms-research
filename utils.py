import mne
import logging
import colorlog
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.model_selection import train_test_split

# =================== Logging ==========================

def get_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s[%(asctime)s] - %(levelname)s - %(message)s'))

    logger = colorlog.getLogger("TMS research")
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# ============== Plotting utilities =====================

def plot_average_response(eeg_data, tmin = -0.05, tmax = 0.2, ymin = None, ymax = None):
    events, event_dict = mne.events_from_annotations(eeg_data)
    epochs = mne.Epochs(eeg_data, events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    average_response_raw = epochs.average()
    average_response_raw.plot(ylim=dict(eeg=[ymin, ymax]))


def plot_average_response_epochs(epochs, ymin=None, ymax=None, tmin=None, tmax=None):
    average_response_raw = epochs.average()
    xlim = (tmin, tmax) if tmin is not None and tmax is not None else None
    average_response_raw.plot(ylim=dict(eeg=[ymin, ymax]), xlim=xlim)


def plot_single_response(eeg_data, channel="Pz", tmin = -0.05, tmax = 0.2):
    events, event_dict = mne.events_from_annotations(eeg_data)
    event_id = event_dict["Stimulus/S  1"]
    epochs = mne.Epochs(eeg_data, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, picks=channel)
    epochs.plot(picks=channel, n_epochs=1, show=True, scalings={'eeg': 50e-6})

# ==========================================================

def plot_epochs_average(epochs, start=-0.05, end=0.25, electrodes=None):
    # Plot the average response of all electrodes in the epochs.
    
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


def plot_spTEP_raw_average(raw, start=-0.05, end=0.25, electrodes=None):
    # Plot the average response of the raw spTEP data.
    
    events, event_dict = mne.events_from_annotations(raw)
    event_id = event_dict["Stimulus/S  1"]
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=start, tmax=end, baseline=None, preload=True)
    plot_epochs_average(epochs, start, end, electrodes)


def plot_rsEEG_raw_average(raw, duration=2, electrodes=None):
    # Plot the average response of the raw rsEEG data.
    
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, events, baseline=None, tmin=0, tmax=duration, preload=True)
    plot_epochs_average(epochs, 0 - (duration / 2), duration / 2, electrodes)


def plot_epochs_average_total(epochs, electrodes=None, start=-0.05, end=0.25, plot_gmfp=True):
    # Generate a plot where all epochs and electrodes have been averaged, resulting in a single response with SEM.

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
        data = epochs.get_data(copy=False)
        average_epoch = np.mean(data, axis=0)        
        gmfa = np.std(average_epoch, axis=0)
        plt.plot(selected_time_points, gmfa[selected_indices], label="GMFA")
    plt.xlabel("Time points")
    plt.ylabel("Mean response")
    plt.legend()
    plt.show()
    
def plot_epochs_gmfa(epochs, start=-0.05, end=0.25):
    # Generate a plot where the grand mean field amplitude (GMFA) is plotted.
    
    data = epochs.get_data(copy=False)
    average_epoch = np.mean(data, axis=0)
    gmfa = np.std(average_epoch, axis=0)
    time_points = np.linspace(-1, 1, data.shape[2])
    plt.plot(time_points, gmfa)
    plt.xlabel("Time points")
    plt.ylabel("GMFA")
    plt.xlim([start, end])
    plt.show()

# ===================================================================================

def get_train_test_split(directory, test_size=0.2, random_state=42):
    """Create a train-test split of all filenames in a directory based on the participant to avoid data leakage."""
    filenames = os.listdir(directory)
    participant_dict = {}
    for filename in filenames:
        match = re.search(r'H_(\d{2})', filename)
        if match:
            participant_num = match.group(1)
            if participant_num not in participant_dict:
                participant_dict[participant_num] = []
            participant_dict[participant_num].append(filename)
    participants = list(participant_dict.keys())
    train_participants, test_participants = train_test_split(participants, test_size=test_size, random_state=random_state)

    train_files = [filename for participant in train_participants for filename in participant_dict[participant]]
    test_files = [filename for participant in test_participants for filename in participant_dict[participant]]

    return train_files, test_files