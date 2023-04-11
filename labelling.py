import scipy.io as scio
import os
import mne
import numpy as np

SUBJECTS_BCI_IV = ['A{0:02d}'.format(i) for i in range(1, 10)]
data_path = "./drive/MyDrive/Data"
result_path = "./drive/MyDrive/results"
session_types = ['T', 'E']

if(not os.path.exists(result_path)):
    os.makedirs(result_path)

for name in SUBJECTS_BCI_IV:
    for session_type in session_types:
        label = scio.loadmat(os.path.join(data_path, name + session_type + '.mat'))['classlabel']
        data = mne.io.read_raw_gdf(os.path.join(data_path, name + session_type + '.gdf'))

        events = mne.events_from_annotations(data)
        epochs = mne.Epochs(data, events[0], event_id=events[1]['768'], tmin=0, tmax=4, baseline=None, detrend=None, preload=True) # Create epochs with start event (code=768) as trigger
        epoched_data = epochs.get_data()

        assert epoched_data.shape[0] == label.shape[0], "Trials and label counts do not match"

        epoched_data = np.nan_to_num(epoched_data)
        epoched_data = epoched_data[:, :22, :1000] # Keep only EEG channels
        epoched_data = np.transpose(epoched_data, (2, 1, 0)) # Expected shape for downstream code

        mat = {'data': epoched_data, 'label': label}
        scio.savemat(os.path.join(result_path, name + session_type + ".mat"), mat)

        print("Finished processing", name + session_type)

print("Finished processing all subjects.")