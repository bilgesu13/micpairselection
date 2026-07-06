import numpy as np
from numpy import linalg as LA

def gen_searchGrid(micPos, dim1, dim2, dim3, mode, c):
    # number of microphones
    M = len(micPos);
    # number of microphone pairs
    P = M * (M - 1) / 2;

    N_dim1 = len(dim1);
    N_dim2 = len(dim2);
    N_dim3 = len(dim3);


    M = len(micPos);
    P = M * (M - 1) / 2;
    rows=N_dim1*N_dim2*N_dim3;
    #rows=N_dim1*N_dim2;
    Delta_t_i = np.zeros((rows,int(P)));
    #Delta_t_i = np.zeros((8101, 15));


    # loc=np.zeros((8101,3));
    loc = np.zeros((rows, 3));
    #loc = np.zeros((rows, 2));

    i = 0;
    for n_dim1 in range (1,N_dim1+1):
        for n_dim2 in range (1,N_dim2+1):
            for n_dim3 in range (1, N_dim3+1):
                #print (n_dim2)
                i = i + 1;
                if mode=='cartesian':
                    x = dim1[n_dim1-1];
                    y = dim2[n_dim2-1];
                    z = dim3[n_dim3-1];
                    # location in cartesian coordinates
                    #loc[i-1,:] = [x, y];
                    loc[i-1,:] = [x, y, z];
                    # TDOA
                    #Delta_t_i[i-1,:] = np.transpose(loc2TDOA(micPos, [x, y], c));
                    Delta_t_i[i-1,:] = np.transpose(loc2TDOA(micPos, [x, y, z], c));
                    #print('hello')
                if mode=='polar':
                    r = dim1[n_dim1];
                    theta = dim2[n_dim2];
                    x = r * np.cos(np.deg2rad(theta));
                    y = r * np.sin(np.deg2rad(theta));
                    # location in cartesian coordinates
                    loc[i-1,:] = x, y;
                    # TDOA
                    Delta_t_i[i-1,:] = loc2TDOA(micPos, [x, y], c);
                if mode=='spherical':
                    ang_pol = dim1[n_dim1-1];
                    ang_az = dim2[n_dim2-1];
                    # location in far field spherical coordinates
                    Delta_t_i[i-1,:], loc[i-1,:] = spher2TDOA(micPos, ang_pol, ang_az, c);
                    print(i)
                    if ang_pol == 0 or ang_pol == 180:
                        break

    Delta_t_i= Delta_t_i[0:i,:];
    loc=loc[0:i,:];
    return(loc, Delta_t_i)

def loc2TDOA(micPos, loc, c):
    M=len(micPos);
    P = M * (M - 1) / 2;
    #propagation time
    Z=micPos - np.tile(loc, (M, 1));
    a=np.sum(np.power(Z, 2),1);
    dist = np.sqrt(a);
    propTime = dist / c;
    propTime = propTime.reshape(-1,1);

    delta_t = np.zeros((int(P), 1));
    p = 0;
    for mprime in range (1,M+1):
        for m in range (mprime+1,M+1):
            p = p + 1;
            delta_t[p-1] = propTime[m-1] - propTime[mprime-1];

    return delta_t;

def spher2TDOA(micPos, ang_pol, ang_az, c):
    M = len(micPos);
    P = M * (M - 1) / 2;

    DOA_vec = [np.sin(np.deg2rad(ang_pol)) * np.cos(np.deg2rad(ang_az)), np.sin(np.deg2rad(ang_pol)) * np.sin(np.deg2rad(ang_az)), np.cos(np.deg2rad(ang_pol))];
    DOA_vec = np.matrix(DOA_vec);

    b = -np.transpose(DOA_vec);
    b = np.matrix(b);
    micPos = np.matrix(micPos);

    delta_t = np.zeros((int(P), 1));
    p = 0;
    for mprime in range (1,M+1):
        for m in range (mprime+1, M+1):
            p = p + 1;

            a = np.transpose(micPos[m-1,:]) - np.transpose(micPos[mprime-1,:]);
            a=np.matrix(a);
            norms=(LA.norm(a)*LA.norm(b))
            delta_t[p-1] = (LA.norm(a) / c) * (np.transpose(a)*b/norms);
    return (np.transpose(delta_t), DOA_vec)