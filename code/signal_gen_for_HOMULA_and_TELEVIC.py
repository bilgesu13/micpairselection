
#%%

# ============================================================
# ===   GENERATE FINAL HOMULA SIGNALS (10 RUNS PER SNR)    ===
# ============================================================

import numpy as np
import scipy.signal as sig
from scipy.io import wavfile

# --- Load clean source ---
fs_src, s_TD_fem = wavfile.read("arctic_a0010.wav")
s_clean = s_TD_fem.astype(float)
s_clean = s_clean / np.max(np.abs(s_clean))

# --- Load RIRs ---
fs_rirs = 48000
rir_data_all = np.load("rir_data_all_homula.npy")
rir_data_all = np.transpose(rir_data_all)  # shape: (num_samples, num_mics)


# --- Resample RIRs ---
fs_target = 16000
rir_data_all = sig.resample_poly(rir_data_all, fs_target, fs_rirs)
fs = fs_target
M = rir_data_all.shape[1]

# --- Convolve clean signal ---
x_clean_full = np.zeros((len(s_clean) + rir_data_all.shape[0] - 1, M))
for mic in range(M):
    x_clean_full[:, mic] = sig.fftconvolve(s_clean, rir_data_all[:, mic])

# Crop region
crop_start = 2000
crop_end   = 10000
x_clean_crop = x_clean_full[crop_start:crop_end, :]

np.save("x_clean_homula_crop.npy", x_clean_crop)

snr_levels = [0, 6, 12]
num_runs = 10

for run in range(num_runs):

    # ---- Fixed-noise per run (fresh noise every run) ----
    np.random.seed(400 + run)   # different noise for each run, reproducible
    noise_base = np.random.randn(*x_clean_crop.shape)

    # Precompute per-mic noise power
    noise_base_power_mic = np.mean(noise_base**2, axis=0)

    for snr_db in snr_levels:

        x_TD_noisy = np.zeros_like(x_clean_crop)
        target_lin = 10**(snr_db / 10)

        # Scale noise per microphone
        for mic in range(M):
            sig_power = np.mean(x_clean_crop[:, mic] ** 2)
            nb_power  = noise_base_power_mic[mic]

            alpha = np.sqrt(sig_power / (nb_power * target_lin))
            x_TD_noisy[:, mic] = x_clean_crop[:, mic] + alpha * noise_base[:, mic]

        # ---- Save ----
        fname = f"x_TD_homula_FINAL__snr{snr_db}dB_run{run}.npy"
        np.save(fname, x_TD_noisy)
        print("Saved:", fname)

print("\n✔️ Done! Generated 10 runs for each SNR level.")

#%%%

# ============================================================
# ===   GENERATE FINAL TELEVIC SIGNALS (10 RUNS PER SNR)    ===
# ============================================================



import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import matplotlib.pyplot as plt


import soundfile as sf


# --- Load clean source ---
fs_src, s_TD_fem = wavfile.read("arctic_a0010.wav")
s_clean = s_TD_fem.astype(float)
s_clean = s_clean / np.max(np.abs(s_clean))

# --- Load and downsample RIRs ---
rir_data_all, fs_rirs = sf.read("src1_rirs.wav")
rir_data_all = rir_data_all[533000:550000, :]

fs_target = 16000
rir_data_all = sig.resample_poly(rir_data_all, fs_target, fs_rirs)
fs = fs_target
plt.plot(rir_data_all[:,0])

M = rir_data_all.shape[1]


# --- Convolve clean signal ---
x_clean_full = np.zeros((len(s_clean) + rir_data_all.shape[0] - 1, M))
for mic in range(M):
    x_clean_full[:, mic] = sig.fftconvolve(s_clean, rir_data_all[:, mic])

# Crop region
crop_start = 2000
crop_end   = 10000
x_clean_crop = x_clean_full[crop_start:crop_end, :]

np.save("x_clean_televic_crop.npy", x_clean_crop)

snr_levels = [0, 6, 12]
num_runs = 20

for run in range(num_runs):

    # ---- Fixed-noise per run (fresh noise every run) ----
    np.random.seed(400 + run)   # different noise for each run, reproducible
    

    noise_base = np.random.randn(*x_clean_crop.shape)

    # Precompute per-mic noise power
    noise_base_power_mic = np.mean(noise_base**2, axis=0)

    for snr_db in snr_levels:

        x_TD_noisy = np.zeros_like(x_clean_crop)
        target_lin = 10**(snr_db / 10)

        # Scale noise per microphone
        for mic in range(M):
            sig_power = np.mean(x_clean_crop[:, mic] ** 2)
            nb_power  = noise_base_power_mic[mic]

            alpha = np.sqrt(sig_power / (nb_power * target_lin))
            x_TD_noisy[:, mic] = x_clean_crop[:, mic] + alpha * noise_base[:, mic]

        # ---- Save ----
        fname = f"x_TD_televic_FINAL__snr{snr_db}dB_run{run}.npy"
        np.save(fname, x_TD_noisy)
        print("Saved:", fname)

print("\n✔️ Done! Generated 10 runs for each SNR level.")