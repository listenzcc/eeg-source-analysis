"""
File: source_analysis.py
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

import numpy as np

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = Path(fs_dir).parent

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = Path(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = Path(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

print(mne.sys_info())

# mne.viz.set_3d_backend('notebook')
mne.viz.get_3d_backend()

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

n_jobs = 8

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
    Set the montage to the standard_1020,
    and convert the sensor's name into upper case.
    '''
    montage = mne.channels.make_standard_montage('standard_1020')

    names = montage.ch_names
    for n in names:
        montage.rename_channels({n: n.upper()})
    montage.rename_channels({'O9': 'CB1', 'O10': 'CB2'})

    raw.set_montage(montage, on_missing='warn')


set_montage()

# %%
# Extract events
events, event_id = mne.events_from_annotations(raw)
print(raw)
print(events)
print(event_id)


# Check that the locations of EEG electrodes is correct with respect to MRI
# Only works in cli mode
# try:
#     fig = mne.viz.plot_alignment(
#         raw.info,
#         src=src,
#         eeg=["original", "projected"],
#         trans=trans,
#         show_axes=True,
#         mri_fiducials=True,
#         interaction='trackball',
#         dig="fiducials",
#     )
#     fig.plot()
# except Exception:
#     pass

# %%
fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, n_jobs=n_jobs)
print(fwd)

# %% ---- 2023-11-07 ------------------------
# Pending
epochs_raw = mne.Epochs(raw, events=events, event_id=event_id,
                        baseline=(None, 0),
                        picks=['eeg'], **epochs_crop)
epochs_raw.set_eeg_reference(projection=True)
epochs = epochs_raw['1']
epochs.load_data()
epochs.filter(**filter_setup)
print(epochs)

evoked = epochs.average()
print(evoked)

fig = mne.viz.plot_evoked_joint(evoked)

# %% ---- 2023-11-07 ------------------------
# Pending
# Covariance matrix
cov = mne.compute_covariance(epochs)
inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov)
stc = mne.minimum_norm.apply_inverse(evoked, inv)
print(cov)
print(inv)
print(stc)
print(np.max(stc.data), np.min(stc.data))

# %%
try:
    stc.plot()
except Exception:
    import traceback
    traceback.print_exc()
    pass

# %%
initial_time = 0.1

stc_fs = mne.compute_source_morph(
    stc, subject, "fsaverage", subjects_dir, smooth=5, verbose="error"
).apply(stc)

brain = stc_fs.plot(
    subjects_dir=subjects_dir,
    initial_time=initial_time,
    clim=dict(kind="value", lims=[2, 3, 5]),
    surface="flat",
    hemi="both",
    size=(1000, 500),
    smoothing_steps=5,
    time_viewer=False,
    add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)),
)

# to help orient us, let's add a parcellation (red=auditory, green=motor,
# blue=visual)
brain.add_annotation("HCPMMP1_combined", borders=2)


# You can save a movie like the one on our documentation website with:
brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16,
                 interpolation='linear', framerate=10)

# %%
# %%
# %%
# %%
