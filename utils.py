import mne
import logging
import colorlog


def get_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s[%(asctime)s] - %(levelname)s - %(message)s'))

    logger = colorlog.getLogger("TMS research")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

# =================================================

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
    
# =================================================


