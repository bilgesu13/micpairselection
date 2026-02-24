
#%%
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import math

# =========================================================
# Settings
# =========================================================
run_indices = range(5, 11)   # 5,6,7,8,9,10
fs = 16000
rt60_tgt = 0.74   # same as HOMULA-like
room_dim = [5, 15]
absorption_coeff = 0.4
scattering_coeff = 0.5

# Microphone coordinates (3D full)
mic_coords_full = np.array([
    [0.97, 4.18, 1.19], [1.78, 4.17, 1.19], [2.57, 4.19, 1.19],
    [3.40, 4.18, 1.19], [4.20, 4.16, 1.19], [0.97, 5.77, 1.19],
    [1.80, 5.77, 1.19], [2.57, 5.76, 1.19], [3.40, 5.76, 1.19],
    [4.19, 5.76, 1.19], [0.99, 7.38, 1.19], [1.79, 7.36, 1.19],
    [2.56, 7.35, 1.19], [3.38, 7.37, 1.19], [4.15, 7.37, 1.19],
    [0.97, 8.98, 1.19], [1.77, 8.97, 1.19], [2.56, 8.98, 1.19],
    [3.39, 8.99, 1.19], [4.17, 8.99, 1.19], [0.96, 10.56, 1.19],
    [1.76, 10.56, 1.19], [2.56, 10.55, 1.19], [3.38, 10.56, 1.19],
    [4.19, 10.57, 1.19]
])
mic_coords_xy = mic_coords_full[:, :2]

# Load all 200 random source positions 
source_coords_all = np.load("200_new_realistic_source_coords.npy")

# STFT parameters
N_STFT = 2048
R_STFT = N_STFT // 2
win = np.sqrt(np.hanning(N_STFT))
N_STFT_half = N_STFT // 2 + 1
pi = math.pi
omega = 2 * pi * np.linspace(0, fs/2, N_STFT_half)

# =========================================================
# Load grid & Δt
# =========================================================
from gen_searchGrid import gen_searchGrid

xg = np.arange(0, 5, 0.05)
yg = np.arange(0, 15, 0.05)
zg = np.array([1.2])

DOAvec_i, Delta_t_i = gen_searchGrid(
    mic_coords_full, xg, yg, zg, 'cartesian', 340
)

# =========================================================
# Helper: Run SRP
# =========================================================
def run_single_simulation(source_coord, snr_db):
    
    # Compute suitable reflection order
    _, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Material
    custom_material = pra.Material(absorption_coeff, scattering=scattering_coeff)

    # Build room
    room = pra.ShoeBox(
        room_dim, fs=fs,
        materials=custom_material,
        max_order=max_order
    )

    # Add mic array
    mic_array = pra.MicrophoneArray(mic_coords_xy.T, fs)
    room.add_microphone_array(mic_array)

    # Load speech
    sr_fs, audio = wavfile.read("arctic_a0010.wav")
    if sr_fs != fs:
        raise ValueError("Speech file sampling frequency mismatch.")

    # Add source
    room.add_source(source_coord[:2], signal=audio)

    # Simulate with noise
    room.simulate(reference_mic=0, snr=snr_db)

    # Signals
    x_TD = room.mic_array.signals.T
    x_TD = x_TD[2000:10000, :]

    # STFT
    from calc_STFT import calc_STFT
    x_STFT, _ = calc_STFT(x_TD, fs, win, N_STFT, R_STFT, 'onesided')
    a = np.mean(x_STFT, axis=1)
    x_STFT = a.reshape((1025, 1, 25))

    # GCC
    from calc_FD_GCC import calc_FD_GCC
    psi_STFT = calc_FD_GCC(x_STFT)

    # SRP
    from calc_SRPconv import calc_SRPconv
    SRP = calc_SRPconv(psi_STFT, omega, Delta_t_i)

    # Estimate DOA/position
    maxIdx = np.argmax(SRP, axis=1).reshape(-1, 1)
    est = DOAvec_i[maxIdx, :]

    # Euclidean error (2D)
    err = source_coord[:2] - est[0, 0, :2]
    err_euc = np.sqrt(np.sum(err**2))

    return err_euc


# =========================================================
# MAIN LOOP FOR 3 SNR LEVELS
# =========================================================
snr_levels = [0, 6, 12]

for snr_db in snr_levels:
    print(f"\n============================")
    print(f" Running Conventional SRP @ {snr_db} dB")
    print(f"============================\n")

    errors = []

    for run_idx in run_indices:
        print(f"Run for source index {run_idx}")
        src = source_coords_all[run_idx]
        errors.append(run_single_simulation(src, snr_db))

    errors = np.array(errors)
    np.save(f"simulations_conv_{snr_db}dB.npy", errors)

    print(f"\nSaved: simulations_conv_{snr_db}dB.npy")
    print(f"Median = {np.median(errors):.3f} m")
    print(f"Mean   = {np.mean(errors):.3f} m")
#%%