import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import math

# =========================================================
# Settings
# =========================================================
num_runs = 5
fs = 16000
room_dim = [5, 15]

# Reverberation setups (RT60 target → absorption, scattering)
reverb_settings = {
    "1.20s": (0.26, 0.40),
    "1.00s": (0.32, 0.55),
    "0.74s": (0.40, 0.50),
    "0.50s": (0.50, 0.60),
    "0.30s": (0.70, 0.70)
}

# Microphone positions
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

# Load random source positions
source_coords_all = np.load("200_new_realistic_source_coords.npy")

# STFT parameters
N_STFT = 2048
R_STFT = N_STFT // 2
win = np.sqrt(np.hanning(N_STFT))
N_STFT_half = N_STFT // 2 + 1
pi = math.pi
omega = 2 * pi * np.linspace(0, fs/2, N_STFT_half)

# =========================================================
# Grid for SRP
# =========================================================
from gen_searchGrid import gen_searchGrid
xg = np.arange(0, 5, 0.05)
yg = np.arange(0, 15, 0.05)
zg = np.array([1.2])

DOAvec_i, Delta_t_i = gen_searchGrid(
    mic_coords_full, xg, yg, zg, 'cartesian', 340
)

# =========================================================
# Helper: Run conventional SRP once
# =========================================================
def run_single_conv(source_coord, absorption, scattering):

    # Material
    material = pra.Material(absorption, scattering=scattering)

    # Compute reflection order for this RT60
    _, max_order = pra.inverse_sabine(1.0, room_dim)

    # Build room
    room = pra.ShoeBox(room_dim,
                       fs=fs,
                       materials=material,
                       max_order=max_order)

    # Add mics
    mic_array = pra.MicrophoneArray(mic_coords_xy.T, fs)
    room.add_microphone_array(mic_array)

    # Load source
    sr_fs, audio = wavfile.read("arctic_a0010.wav")
    if sr_fs != fs:
        raise ValueError("Sampling mismatch.")
    room.add_source(source_coord[:2], signal=audio)

    # Simulate @ 6 dB
    room.simulate(reference_mic=0, snr=6)

    # Crop mic signals
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

    # Estimate location
    maxIdx = np.argmax(SRP, axis=1).reshape(-1,1)
    est = DOAvec_i[maxIdx, :]

    # Euclidean 2D error
    err = source_coord[:2] - est[0,0,:2]
    return np.sqrt(np.sum(err**2))


# =========================================================
# MAIN EXPERIMENT ACROSS RT60 VALUES
# =========================================================
for label, (absorb, scatter) in reverb_settings.items():

    print(f"\n=======================================")
    print(f"  Running RT60 = {label}   (SNR = 6 dB)")
    print(f"=======================================\n")

    errors = []

    for r in range(num_runs):
        print(f"Run {r+1}/{num_runs}")
        err = run_single_conv(source_coords_all[r], absorb, scatter)
        errors.append(err)

    errors = np.array(errors)
    np.save(f"simulations_conv_RT60_{label}.npy", errors)

    print(f"\nSaved: simulations_conv_RT60_{label}.npy")
    print(f"Median error = {np.median(errors):.3f} m")
    print(f"Mean error   = {np.mean(errors):.3f} m")
