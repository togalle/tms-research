import mne

def plot_average_response(eeg_data, tmin = -0.05, tmax = 0.2):
    events, event_dict = mne.events_from_annotations(eeg_data)
    epochs = mne.Epochs(eeg_data, events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    average_response_raw = epochs.average()
    average_response_raw.plot()
    
def plot_single_response(eeg_data, channel="Pz", tmin = -0.05, tmax = 0.2):
    events, event_dict = mne.events_from_annotations(eeg_data)
    event_id = event_dict["Stimulus/S  1"]
    epochs = mne.Epochs(eeg_data, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, picks=channel)
    epochs.plot(picks=channel, n_epochs=1, show=True, scalings={'eeg': 50e-6})