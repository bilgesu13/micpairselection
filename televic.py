import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
from IPython.display import Audio

num_runs = 20
err_euclidean_all = []

for r in range(num_runs):

    print(f"\n================ RUN {r+1}/{num_runs} ================\n")

    # Load corresponding x_TD file
    # expected filename:
    #x_TD = np.load(f"x_TD_televic_FINAL__snr12dB_run{r}.npy")
    x_TD = np.load(f"x_TD_televic_FINAL20__snr0dB_run{r}.npy")


    fs = 16000

    stepsize=0.05 #0.05; #0.1
    total_error = 0;

    x = np.arange(-3, 3, stepsize).tolist(); #0,5,0.2 #2,5,0.1
    x = np.array(x);
    y = np.arange(-6, 1, stepsize).tolist(); 
    y = np.array(y);
    z = [1.1]; 
    z = np.array(z);


    mic_coords = np.load('mic_coord_televic_Oct.npy') # mic coordinates from Televic dataset


    from gen_searchGrid import gen_searchGrid
    c = 340
    DOAvec_i, Delta_t_i = gen_searchGrid(mic_coords, x, y, z, 'cartesian', c); 


    def sparsity_measure(x):
        l1_norm = np.linalg.norm(x, ord=1)  # L1 norm
        l2_norm = np.linalg.norm(x)  # L2 norm

        lengthL = len(x)

        sparsity = (lengthL/(lengthL-np.sqrt(lengthL)))*(1 - (l1_norm / (np.sqrt(lengthL) * l2_norm)))

        return sparsity

    def xcorr(xi, xj, tau):

        xi = np.array(xi)
        xj = np.array(xj)
        
        n = len(xi)
        cross_corr = []
        
        # Calculate cross-correlation for lags from -tau to +tau
        for lag in range(-tau, tau + 1):
            if lag < 0:
                # Shift xj to the left
                shifted_xj = np.roll(xj, lag)[:n]
            else:
                # Shift xj to the right
                shifted_xj = np.roll(xj, lag)[:n]

            # Compute the dot product for the current lag
            corr_value = np.dot(xi, shifted_xj)
            cross_corr.append(corr_value)
        
        return np.array(cross_corr)

    def distcorr_admm(yi, yj, tau1, tau2, tau, M, rho, k_max, eps_abs, eps_rel, r_jj_peak, r_ii_peak,Yj, tildeYi, Yi, tildeYj):
        # Initialization of internal ADMM variables
        hat_r_ij = np.zeros(tau1 + tau2 + 1)  # Consensus variable initialization
        hat_lambda_ij_j = np.zeros(tau1 + tau2 + 1)  # Node j dual variable initialization
        hat_lambda_ij_i = np.zeros(tau1 + tau2 + 1)  # Node i dual variable initialization

        norm_p_k = np.inf  # Primal residual norm (iteration k)
        norm_d_k = np.inf  # Dual residual norm (iteration k)
        eps_p = 0; # primal feasibility tolerance
        eps_d = 0; # dual feasibility tolerance

        # Calculation of iteration-independent variables
        t1 = 1
        t2 = len(yi)

        if len(yj) != t2:
            raise ValueError("Signals yi and yj should have the same length")

        r_ii = xcorr(yi, yi, tau); # sample autocorrelation function (SACF)
        r_jj = xcorr(yj, yj, tau);

        s_ijj = np.dot((Yi), r_jj)  # auxiliary signal
        s_iij = np.dot((tildeYj), r_ii) # second auxiliary signal


        s_ijjM = s_ijj[tau1 + 1: tau1 + M]
        s_iijM = s_iij[tau1 + 1: tau1 + M]
        
        YjM = Yj[tau1 + 1: tau1 + M]
        tildeYiM = tildeYi[tau1 + 1: tau1 + M]

        sigma_jM = np.dot(YjM.T, s_ijjM)
        sigma_iM = np.dot(tildeYiM.T, s_iijM)

        # Adding regularization and Cholesky decomposition
        Phi_jM = np.dot(YjM.T, YjM) + rho * np.eye(tau1 + tau2 + 1)
        tildePhi_iM = np.dot(tildeYiM.T, tildeYiM) + rho * np.eye(tau1 + tau2 + 1)

        LjM = cholesky(Phi_jM, lower=True)
        tildeLiM = cholesky(tildePhi_iM, lower=True)

        # ADMM algorithm
        k = 1  # Initialization of iteration index
        while not (norm_p_k < eps_p and norm_d_k < eps_d) and (k <= k_max):

            # Solve primal problem at node j
            temp_j = np.linalg.solve(LjM, sigma_jM + rho * hat_r_ij - hat_lambda_ij_j)
            hat_r_ij_j = np.linalg.solve(LjM.T, temp_j)

            # Solve primal problem at node i
            temp_i = np.linalg.solve(tildeLiM, sigma_iM + rho * hat_r_ij - hat_lambda_ij_i)
            hat_r_ij_i = np.linalg.solve(tildeLiM.T, temp_i)

            # compute consensus variable at node i and j 
            hat_r_ij_old = hat_r_ij
            hat_r_ij = (hat_r_ij_j + hat_r_ij_i) / 2  # Consensus variable at node i and j

            # Solve dual problem at node j
            hat_lambda_ij_j += rho * (hat_r_ij_j - hat_r_ij)
            norm_hat_lambda_ij_j = np.linalg.norm(hat_lambda_ij_j);   
            # Solve dual problem at node i
            hat_lambda_ij_i += rho * (hat_r_ij_i - hat_r_ij)
            norm_hat_lambda_ij_i = np.linalg.norm(hat_lambda_ij_i);   


            norm_p_k = np.linalg.norm(np.concatenate((hat_r_ij_j - hat_r_ij,hat_r_ij_i - hat_r_ij), axis = 0)); # primal residual norm
            norm_d_k = np.sqrt(2)*rho*np.linalg.norm(hat_r_ij - hat_r_ij_old);           # dual residual norm
            eps_p = eps_abs + eps_rel*max(int(np.linalg.norm(np.concatenate((hat_r_ij_j,hat_r_ij_i)), axis = 0)),int(np.linalg.norm((np.concatenate((hat_r_ij,hat_r_ij),axis = 0))))); # primal feasibility tolerance
            eps_d = eps_abs + eps_rel*np.sqrt(norm_hat_lambda_ij_j**2 + norm_hat_lambda_ij_i**2);        # dual feasibility tolerance

            k = k+1;        
            #print(k)
        return hat_r_ij,norm_p_k,norm_d_k,k,s_iijM,s_ijjM


    #### distcorr for correlation at lag 0 only
    from scipy.linalg import cholesky
    def distcorr_admm_lag0(yi, yj, M, rho, k_max, eps_abs, eps_rel):
        # Initialization of internal ADMM variables
        hat_r_ij = 0  # Consensus variable initialization
        hat_lambda_ij_j = 0 # Node j dual variable initialization
        hat_lambda_ij_i = 0  # Node i dual variable initialization
        norm_p_k = np.inf  # Primal residual norm (iteration k)
        norm_d_k = np.inf  # Dual residual norm (iteration k)
        eps_p = 0; # primal feasibility tolerance
        eps_d = 0; # dual feasibility tolerance

        # Calculation of iteration-independent variables
        t1 = 1
        t2 = len(yi)

        if len(yj) != t2:
            raise ValueError("Signals yi and yj should have the same length")

    # Autocorrelations (at lag 0)
        r_ii = np.dot(yi, yi)  # Autocorrelation of yi at lag 0
        r_jj = np.dot(yj, yj)  # Autocorrelation of yj at lag 0

    # Auxiliary variables for the ADMM steps
        s_ijj = r_jj * yi[:M]  # auxiliary signal for node i
        s_iij = r_ii * yj[:M]  # auxiliary signal for node j

        # Define auxiliary variables for ADMM updates
        sigma_jM = np.dot(yj[:M], s_ijj)
        sigma_iM = np.dot(yi[:M], s_iij)

        # Adding regularization and Cholesky decomposition for lag 0
        Phi_jM = np.dot(yj.T, yj) + rho
        tildePhi_iM = np.dot(yi.T, yi) + rho

        LjM = np.sqrt(Phi_jM)
        tildeLiM = np.sqrt(tildePhi_iM)

        # ADMM algorithm at lag 0
        k = 1  # Initialization of iteration index
        while not (norm_p_k < eps_p and norm_d_k < eps_d) and (k <= k_max):
            # Solve primal problem at node j
            temp_j = (sigma_jM + rho * hat_r_ij - hat_lambda_ij_j) / LjM
            hat_r_ij_j = temp_j / LjM

            # Solve primal problem at node i
            temp_i = (sigma_iM + rho * hat_r_ij - hat_lambda_ij_i) / tildeLiM
            hat_r_ij_i = temp_i / tildeLiM

            # Compute consensus variable
            hat_r_ij_old = hat_r_ij
            hat_r_ij = (hat_r_ij_j + hat_r_ij_i) / 2

            # Update dual variables
            hat_lambda_ij_j += rho * (hat_r_ij_j - hat_r_ij)
            hat_lambda_ij_i += rho * (hat_r_ij_i - hat_r_ij)

            # Calculate residual norms
            norm_p_k = np.abs(hat_r_ij_j - hat_r_ij) + np.abs(hat_r_ij_i - hat_r_ij)
            norm_d_k = np.sqrt(2) * rho * np.abs(hat_r_ij - hat_r_ij_old)

            # Update tolerances
            eps_p = eps_abs + eps_rel * max(np.abs(hat_r_ij_j), np.abs(hat_r_ij_i), np.abs(hat_r_ij))
            eps_d = eps_abs + eps_rel * np.sqrt(hat_lambda_ij_j ** 2 + hat_lambda_ij_i ** 2)

            k += 1

        return hat_r_ij, norm_p_k, norm_d_k, k

    ### source coordinate from recovering_coordinates_again.py
    source_coords = np.load('src_coord_televic_Oct.npy')
    print("source coords:", source_coords)

    num_runs = 1

    for runs in range(num_runs):

        from itertools import combinations
        from scipy.signal import correlate
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.linalg import toeplitz, hankel, cholesky

        N = len(x_TD)

        microphone_combinations = list(combinations(range(23), 2))

        cross_correlations = []

        Yi = []
        Yj = []
        tildeYi = []
        tildeYj = []

        #M = 16; # 32 64 128 256 512 1024 2048

        # Compute cross-correlations for each microphone pair
        for mic1, mic2 in microphone_combinations:
            # -- cross corr used before in Eusipco paper
            #cross_corr = correlate(x_TD[:, mic1], x_TD[:, mic2], mode='full')
            
            # -- Compute cross-correlation with a given tau
            tau = 20 

            t1 = 1;
            t2 = N;
            tau1 = 100;
            tau2 = 100;
            tau = 100;

            # Toeplitz matrices
            # Construct the column and row for Toeplitz matrix
            toeplitz_col = np.concatenate([x_TD[:, mic1][t1-1 + tau1:t2], np.zeros(tau1)])
            toeplitz_row = np.concatenate([x_TD[:, mic1][t1 + tau1:t1-1:-1], np.zeros(tau2)])

            Yi = toeplitz(toeplitz_col, toeplitz_row)  # Toeplitz matrix for yi

            toeplitz_col = np.concatenate([x_TD[:, mic2][t1-1 + tau1:t2], np.zeros(tau1)])
            toeplitz_row = np.concatenate([x_TD[:, mic2][t1 + tau1:t1-1:-1], np.zeros(tau2)])

            Yj = toeplitz(toeplitz_col, toeplitz_row)  # Toeplitz matrix for yj

            # Hankel matrices
            # First column with 'tau1' zeros, then signal from 't1' to 't2 - tau1'
            hankel_col = np.concatenate([np.zeros(tau1), x_TD[:, mic1][t1 - 1:t2 - tau1]])
            # Last row from 't2 - tau1' to 't2', followed by 'tau2' zeros
            hankel_row = np.concatenate([x_TD[:, mic1][t2 - tau1 - 1:t2], np.zeros(tau2)])

            tildeYi = hankel(hankel_col, hankel_row)  # Hankel matrix for yi

            # Similarly for yj
            hankel_col = np.concatenate([np.zeros(tau1), x_TD[:, mic2][t1 - 1:t2 - tau1]])
            hankel_row = np.concatenate([x_TD[:, mic2][t2 - tau1 - 1:t2], np.zeros(tau2)])

            tildeYj = hankel(hankel_col, hankel_row)  # Hankel matrix for yj

            r_ii = xcorr(x_TD[:, mic1], x_TD[:, mic1], tau); # sample autocorrelation function of yi
            r_jj = xcorr(x_TD[:, mic2], x_TD[:, mic2], tau); # sample autocorrelation function of yj

            r_ii_peak = np.argmax(r_ii)
            r_jj_peak = np.argmax(r_jj)

            s_ijj = np.dot((Yi), r_jj) # auxiliary signal computed in node i by filtering the entire yi with noncasual filter rjj
            s_iij = np.dot((tildeYj), r_ii) # second auxiliary signal ...

            # -- Proposed algorithm 3: consensus deconvolution (ADMM) %%

            rho = 0.01;     # ADMM parameter
            #rho = 1;
            # the optimal value for the ADMM parameter rho depends on the SNR:
            # (1) SNR = 0 >>> rho = 1
            # (2) SNR = 10 >>> rho = 0.2
            # (3) SNR = Inf >>> rho = 0.01
            eps_abs = 0;    # absolute error tolerance for ADMM termination criterion
            eps_rel = 1e-4; # relative error tolerance for ADMM termination criterion
        
            M = 16
            #k = 100; 
            k = 10; 

            #cross_corr, norm_p, norm_d, k, s_iijM,s_ijjM = distcorr_admm(x_TD[:, mic1], x_TD[:, mic2], tau1, tau2, tau, M, rho, k, eps_abs, eps_rel,r_jj_peak,r_ii_peak,Yj, tildeYi, Yi, tildeYj)
            cross_corr, norm_p, norm_d, k = distcorr_admm_lag0(x_TD[:, mic1], x_TD[:, mic2], M, rho, k, eps_abs, eps_rel)
            cross_correlations.append(cross_corr)


        # Create a time axis in seconds
        #time_axis = np.arange(-len(x_TD)+1, len(x_TD)) / f s
        time_axis = np.arange(-tau, tau + 1)

        cross_correlations = np.array(cross_correlations)

        #### compute sparsity
        from scipy.stats import kurtosis
        # Create all combinations of microphone pairs
        microphone_combinations = list(combinations(range(23), 2))

        results = np.zeros(len(microphone_combinations))
        # Compute sparsity measure for each microphone pair
        for i in range(len(microphone_combinations)):
            results[i] = sparsity_measure(cross_correlations[i,:]) #norm based sparsity measure
            #results[i] = kurtosis(cross_correlations[i,:]) #kurtosis based sparsity measure
            mic1, mic2 = microphone_combinations[i]
            #print(f"Sparsity Measure: Mic {mic1} vs. Mic {mic2}", results[i])


        ### now cross_correlations have the estimated cross correlations at lag 0, therefore directly threshold them to
        ### and select broadside microphone pairs
        cross_correlations = np.array(cross_correlations)
        results = cross_correlations

        ### calculate the threshold based on gossip
        import random

        class Node:
            def __init__(self, node_id, initial_value):
                self.node_id = node_id
                self.value = initial_value
                self.neighbors = []  # To store neighboring nodes
                self.top_values = [initial_value]  # Keep track of top values

            
            def add_neighbor(self, neighbor_node):
                self.neighbors.append(neighbor_node)

            def push_pull_gossip(self):
                if self.neighbors:
                    neighbor = random.choice(self.neighbors)
                    
                    # Push local value to neighbor and pull neighbor's top values
                    local_top_values = self.top_values
                    neighbor_top_values = neighbor.top_values
                    
                    # Merge and keep the top values (assumes sorted lists)
                    combined_top_values = sorted(set(local_top_values + neighbor_top_values), reverse=True)
                    self.top_values = combined_top_values[:6]  # Keep only the top 6 values
                    neighbor.top_values = self.top_values  # Update neighbor's top values

        # create a random network topology for gossip 
        def create_random_network(nodes, max_neighbors=5):
            for node in nodes:
                num_neighbors = random.randint(1, max_neighbors)  # Assign random number of neighbors
                possible_neighbors = [n for n in nodes if n.node_id != node.node_id]
                node.neighbors = random.sample(possible_neighbors, num_neighbors)

        # Run gossip protocol for a certain number of rounds
        def run_gossip_protocol(nodes, rounds):
            for _ in range(rounds):
                for node in nodes:
                    node.push_pull_gossip()

        # Create 253 nodes
        # Initialize 253 nodes with values from the 'results' array (23 microphones so 253 pairs)
        nodes = [Node(i, results[i]) for i in range(253)]

        # Create random neighbors between nodes
        create_random_network(nodes, max_neighbors=5)

        # Run gossip for a specified number of rounds
        rounds = 10  # You can adjust the number of rounds as needed
        run_gossip_protocol(nodes, rounds)


        # After gossip, gather all top values from each node
        all_top_values = set()
        for node in nodes:
            all_top_values.update(node.top_values)

        # Select the top 6 unique values
        final_top_values = sorted(all_top_values, reverse=True)[:6]

        print("Top 6 values after gossip:", final_top_values)


        above_threshold_indices = [np.where(results == value)[0][0] for value in final_top_values]

        print("Above threshold indices", above_threshold_indices)

        stepsize=0.05 #0.05; #0.1
        x = np.arange(-3, 3, stepsize).tolist(); #0,5,0.2 #2,5,0.1
        x = np.array(x);
        y = np.arange(-6, 1, stepsize).tolist(); 
        y = np.array(y);
        z = [1.1]; 
        z = np.array(z);

        import math
        pi=math.pi
        w_0 = pi*fs;
        # STFT PARAMETERS
        # window size
        N_STFT = 2048;
        #N_STFT = 256;
        # shift
        R_STFT = N_STFT/2;
        # window
        win = np.sqrt(np.hanning(N_STFT));
        N_STFT_half = math.floor(N_STFT/2)+1;
        # frequency vector
        omega = 2*pi*np.transpose(np.linspace(0,fs/2,N_STFT_half));


        # transform to STFT domain
        from calc_STFT import calc_STFT
        x_STFT,f_x = calc_STFT(x_TD, fs, win, N_STFT, R_STFT, 'onesided');

        a = np.mean(x_STFT,1) 

        num_mics = 23
        a = np.reshape(a,(1025,1,num_mics))

        #a = np.reshape(a,(129,1,num_mics))

        x_STFT = a

        from calc_FD_GCC import calc_FD_GCC
        psi_STFT = calc_FD_GCC(x_STFT);

        ### random selection of microphone pairs - uncomment below if random ###
        # num_to_select = 6
        # random_indices = np.random.choice(len(microphone_combinations), size=num_to_select, replace=False)
        # #random_indices = [170, 213, 231, 273, 161, 57] # example for the figure in paper
        # above_threshold_indices = list(random_indices)
        # print("above threshold indices", above_threshold_indices)

        non_zero_elements = psi_STFT[:,:,above_threshold_indices];
        D = Delta_t_i[:,above_threshold_indices];
        
        from calc_SRPconv import calc_SRPconv
        #SRP_conv = calc_SRPconv(psi_STFT, omega, Delta_t_i); ## conventional SRP
        SRP_conv = calc_SRPconv(non_zero_elements, omega, D);

        maxIdx_conv = np.argmax(SRP_conv, 1);
        maxIdx_conv = maxIdx_conv.reshape(-1, 1);
        estim_DOAvec = DOAvec_i[maxIdx_conv,:];

        
        import matplotlib
        stepsize = 0.05 #0.05;
        source_coord = source_coords

        colors = ['magenta', 'navy', 'pink', 'purple', 'orange', 'cyan','black','white','green']

        markers = ['o', '>', 'v', 'd', 's', '<', 'p','s','v']

        x = np.arange(-3, 3, stepsize).tolist(); #0,5,0.2 #2,5,0.1
        x = np.array(x);
        y = np.arange(-6, 1, stepsize).tolist(); 
        y = np.array(y);


        SRP_conv_graph = SRP_conv.reshape(len(x), len(y));
        #fig = plt.figure(figsize=(18, 18))
        stepsize = 1

        plt.figure(figsize=(8, 6))  # Adjust the figure size (width, height) in inches
        plt.imshow(np.transpose(SRP_conv_graph),origin='lower',extent = [-3, 3, -6, 1])
        #plt.imshow(np.transpose(SRP_conv_graph),origin='lower',extent = [0, 5, 0, 15],cmap='jet')

        plt.colorbar(label='Steered Response Power')

        plt.scatter(source_coord[0]/stepsize,source_coord[1]/stepsize, c='r',s=200,marker = '*')
        plt.scatter(estim_DOAvec[0,0,0]/stepsize,estim_DOAvec[0,0,1]/stepsize,c='b',marker = 'x', s=500)

        plt.scatter(mic_coords[:,0], mic_coords[:,1], c='black', marker = 'o', s=100)


        #plt.scatter(x_coords, y_coords, c='blue', marker='o') ### for the conventional SRP
        all_pairs = []

        for i, pair_number in enumerate(above_threshold_indices):

            pair = list(combinations(range(23), 2))[pair_number]
            all_pairs.append(pair)

            # Extracting coordinates for the chosen pair
            mic1_coords = mic_coords[pair[0]]
            mic2_coords = mic_coords[pair[1]]

            marker = markers[i]
            color = colors[i]
            plt.scatter([mic1_coords[0], mic2_coords[0]], [mic1_coords[1], mic2_coords[1]], marker=marker, color= color, facecolor = 'none', linewidth =2)
            
            plt.plot([mic1_coords[0], mic2_coords[0]], [mic1_coords[1], mic2_coords[1]], linestyle='--', color=color, linewidth=2)

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')

        plt.grid(False)
        plt.show()

        err = source_coord - estim_DOAvec
        err_euclidean = np.sqrt(np.sum(err**2)) #euclidean distance
        print("err euclidean:",err_euclidean)

        total_error += err_euclidean  # Add the current error to the total error
        err_euclidean_all.append(err_euclidean)

    print("total error:", total_error)


err_euclidean_all = np.array(err_euclidean_all)
mean_err = np.mean(err_euclidean_all)
median_err = np.median(err_euclidean_all)
print("\n===================================")
print("  Localization Error Statistics")
print("=====================================")
print(f"Mean error over 100 runs:   {mean_err:.3f} m")
print(f"Median error over 100 runs: {median_err:.3f} m")

# Save errors for later analysis
#np.save("loc_errors_broadside_50runs_televic_12dB.npy", err_euclidean_all)

#print("\nSaved all 50 errors to loc_errors_broadside_50runs_televic_12dB.npy")

np.save("loc_errors_FINAL20___televic_broadside_0dB.npy", err_euclidean_all)
