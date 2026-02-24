# micpairselection
Pair selection code

This repository contains the simulation code used to evaluate microphone-pair selection strategies for efficient Steered Response Power (SRP) localization in reverberant environments. 

The first script is called `main_simulations_sparsity_based_change_rev_or_SNR.py`. In this file, you can change the reverberation time, SNR, and the sparsity metric used for microphone-pair selection. 

The second script is called `main_simulations_zero_lag_based.py` and it performs microphone-pair selection using only the zero-lag cross-correlation measure, which makes it significantly simpler and communication-efficient.

In `main_simulations_sparsity_based_change_rev_or_SNR.py`, you can modify the reverberation characteristics of the simulated room by changing the two parameters `absorption_coeff` and `scattering_coeff`. These control the RT60. For example, if you want an RT60 around 0.74 seconds, set `absorption_coeff = 0.4` and `scattering_coeff = 0.5`.
You can choose any values depending on the environment you wish to simulate. You can also change the SNR. This is done through the line `room.simulate(reference_mic=0, snr=24)`. Changing the number (24, 12, 6, 0, etc.) will directly modify the SNR of the simulated microphone recordings. In the same file, you can change the sparsity metric used to select microphone pairs. The default implementation uses kurtosis: `results[i] = kurtosis(cross_correlations[i,:])`. If you prefer to use another sparsity measure, for example the L1/L2-based sparsity function, you may replace this line with `results[i] = sparsity_measure(cross_correlations[i,:])`.

The second script, `main_simulations_zero_lag_based.py`, implements the Zero-Lag Cross-Correlation Measure. This method uses the cross-correlation at lag zero for each microphone pair, making the selection process very fast. This script operates using the same simulated environment as the first script (same geometry, same adjustable RT60 and SNR, same number of Monte Carlo runs). The only difference is the selection rule.

Both scripts require the following input files placed in the same directory: `arctic_a0010.wav`, which is the clean male speech signal used as the source, and `200_new_realistic_source_coords.npy`, which contains 200 random source positions for Monte Carlo testing. The repository also relies on several helper modules: `gen_searchGrid.py`, `calc_STFT.py`, `calc_FD_GCC.py`, and `calc_SRPconv.py`. These must be present in the repository for the simulations to run. To install the dependencies for the simulations, install PyRoomAcoustics and the scientific Python stack with: `pip install numpy scipy matplotlib pyroomacoustics`.

To run the experiments, simply execute one of the scripts. For example: `python main_simulations_sparsity_based_change_rev_or_SNR.py` or `python main_simulations_zero_lag_based.py`. When the simulation finishes, you will obtain `.npy` files that contain the Euclidean localization error for each of the Monte Carlo runs. The filename reflects the conditions, e.g., `measure2200newsource_realistic_snr24.npy`. These files can be renamed by editing the `np.save(...)` line at the bottom of each script.

The output includes the localization error for every run, the total accumulated error, and (if the plotting code is enabled) SRP heatmaps and other visualizations. Diagnostic information such as the run number and the measured reverberation time is printed during execution.

The script `signal_gen_for_HOMULA_and_Televic.py` generates all microphone signals used for the evaluation with real RIRs. It loads the clean source waveform, normalizes it, loads the real RIRs from the HOMULA dataset or Televic measurements, resamples them to 16 kHz, and performs convolution to obtain microphone recordings. 
It then applies noise at the required SNR levels using controlled scaling per microphone, ensuring reproducibility. The output files are saved with names such as `x_TD_homula_FINAL__snr6dB_run3.npy` or `x_TD_televic_FINAL__snr12dB_run7.npy`. These files are the input for the selection and SRP evaluation experiments in the other scripts. 
For the HOMULA configuration, use homula.py, which includes the HOMULA microphone coordinates (mic_coord_homula.npy), the predefined source coordinates for the experiments, and the HOMULA RIR file (rir_data_all_homula.npy).
For the Televic measurements, use televic.py, together with mic_coord_televic_Oct.npy, src_coord_televic_Oct.npy, and the measured Televic RIRs (src1_rirs.wav).

The repository also includes a script named `plots.py`, which generates all figures used in the paper. This script loads the `.npy` files produced by the simulation scripts and produces accuracy curves, SNR-dependent results, reverberation-dependent results, and grouped boxplots for different microphone pair counts and noise conditions. It uses consistent styling for publication-quality figures and saves each result (e.g., `accuracy_vs_pairs.png`, `accuracy_vs_snr.png`, `boxplot_reverb_localization_error.png`, and others). 
All corresponding `.npy` files used to generate these figures are provided in the repository, ensuring that every plot in the paper can be reproduced directly by running this file.


This repository contains no open-source license file and is intended purely for academic reproducibility. Any Televic measurement data included in the repository remains the property of Televic Conference Systems and may not be redistributed.
The simulation environment reproduced here matches the conditions described in the accompanying manuscript, which is currently under review. 
If you use this code in your research, please cite the manuscript once it becomes publicly available.
