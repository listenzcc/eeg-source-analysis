"""
File: main.py
Author: Chuncheng Zhang
Date: 2023-11-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-11-07 ------------------------
# Requirements and constants
from util import *

import os.path as op
import numpy as np

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

print(mne.sys_info())

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")


# %% ---- 2023-11-07 ------------------------
# Function and class
channel_types_mapping = dict(
    HEO='eog',
    VEO='eog'
)

epochs_crop = dict(
    tmin=-0.2,
    tmax=1.0
)

filter_setup = dict(
    l_freq=0.1,
    h_freq=8.0,
    n_jobs=16
)

# Events
# 16 序列开始，20 序列结束，1 目标，2 非目标，3 被试发现目标按键


# %% ---- 2023-11-07 ------------------------
# Play ground
cnts = [e for e in PPATH.data.iterdir() if e.name.endswith('.cnt')]
print(cnts)

cnt_file = cnts[3]
# cnt_file = cnts[0]
print(cnt_file)
raw = mne.io.read_raw_cnt(cnt_file)

try:
    raw.set_channel_types(channel_types_mapping)
except Exception:
    pass


def set_montage():
    '''
    Set the montage to the standard_1020
    '''
    # mne.channels.get_builtin_montages()
    montage = mne.channels.make_standard_montage('standard_1020')
    # print(montage)

    names = montage.ch_names
    for n in names:
        # print(n, n.upper())
        montage.rename_channels({n: n.upper()})
    montage.rename_channels({'O9': 'CB1', 'O10': 'CB2'})

    # print(raw.ch_names)
    # print(montage.ch_names)

    raw.set_montage(montage, on_missing='warn')


set_montage()

# %%
# Extract events
events, event_id = mne.events_from_annotations(raw)
print(raw)
print(events)
print(event_id)
fig = mne.viz.plot_events(events, event_id=event_id, show=False)
fig = raw.plot_sensors(show_names=True, show=False)

# %%
# ICA de-noise
ica = mne.preprocessing.ICA(n_components=20)
ica.fit(raw)
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude += eog_indices
raw.load_data()
ica.apply(raw)


# %% ---- 2023-11-07 ------------------------
# Pending
epochs = mne.Epochs(raw, events=events, event_id=event_id,
                    baseline=(None, 0),
                    picks=['eeg'], **epochs_crop)
epochs = epochs['1']
epochs.load_data()
epochs.filter(**filter_setup)
print(epochs)

evoked = epochs.average()
print(evoked)

fig = mne.viz.plot_evoked(evoked, spatial_colors=True)

# %% ---- 2023-11-07 ------------------------
# Pending

# %%
