
#=======================================================================================================================
#=======================================================================================================================
# Package Importing
import numpy as np
from generatorFun_pt import *
import h5py
import time

def K_nearest(h_true_smp, h_fake_smp, rx_num, tx_num, delay_num, flag):
    t1 = time.time()

    h_true = np.reshape(h_true_smp, [h_true_smp.shape[0], rx_num * tx_num * delay_num])
    h_fake = np.reshape(h_fake_smp, [h_fake_smp.shape[0], rx_num * tx_num * delay_num])
    h_true_norm = np.linalg.norm(h_true, axis=1)
    h_fake_norm = np.linalg.norm(h_fake, axis=1)
    h_true_norm = h_true_norm[:, np.newaxis]
    h_fake_norm = h_fake_norm[:, np.newaxis]
    h_true_norm_matrix = np.tile(h_true_norm, (1, rx_num * tx_num * delay_num))
    h_fake_norm_matrix = np.tile(h_fake_norm, (1, rx_num * tx_num * delay_num))
    h_true = h_true / h_true_norm_matrix
    h_fake = h_fake / h_fake_norm_matrix

    t2 = time.time()

    r_s = abs(np.dot(h_fake, h_true.conj().T))
    r = r_s * r_s

    t3 = time.time()

    r_max = np.max(r, axis = 1)
    r_idx = np.argmax(r, axis = 1)
    K_sim_abs_mean = np.mean(r_max)

    counts_idx, counts_num = np.unique(r_idx,return_counts=True)
    K_multi = np.zeros((1, h_fake_smp.shape[0]))
    K_multi[:, counts_idx] = counts_num
    K_multi_std = float(np.sqrt(np.var(K_multi, axis=1) * h_fake_smp.shape[0] / (h_fake_smp.shape[0] - 1)))

    t4 = time.time()

    print('Norming: ' + str(t2 - t1) + 's')
    print('Matrix multiplication: ' + str(t3 - t2) + 's')
    print('UE Loop: ' + str(t4 - t3) + 's')
    return K_sim_abs_mean, K_multi_std, K_multi_std / K_sim_abs_mean
#=======================================================================================================================
#=======================================================================================================================
t1 = time.time()
# Parameter Setting
NUM_RX = 4
NUM_TX = 32
NUM_DELAY = 32
NUM_REAL_1 = 500
NUM_REAL_2 = 4000
NUM_FAKE_1 = NUM_REAL_1
NUM_FAKE_2 = NUM_REAL_2
#=======================================================================================================================
#=======================================================================================================================
# Data 1 Loading
real_1_test = h5py.File('H1_32T4R.mat', 'r')
real_1_test = np.transpose(real_1_test['H1_32T4R'][:])
real_1_test = real_1_test[::int(real_1_test.shape[0] / NUM_REAL_1), :, :, :]
real_1_test = real_1_test['real'] + real_1_test['imag'] * 1j
# Data 2 Loading
real_2_test = h5py.File('H2_32T4R.mat', 'r')
real_2_test = np.transpose(real_2_test['H2_32T4R'][:])
real_2_test = real_2_test[::int(real_2_test.shape[0] / NUM_REAL_2), :, :, :]
real_2_test = real_2_test['real'] + real_2_test['imag'] * 1j
t2 = time.time()
#=======================================================================================================================
#=======================================================================================================================
# Data 1 Generating
fake_1 = generator_1(NUM_FAKE_1, 'generator_1.pth.tar', 'H1_32T4R.mat')
# Data 2 Generating
fake_2 = generator_2(NUM_FAKE_2, 'generator_2.pth.tar', 'H2_32T4R.mat')
t3 = time.time()
#=======================================================================================================================
#=======================================================================================================================
# Data checking
if (np.shape(fake_1) == np.array([NUM_FAKE_1, NUM_RX, NUM_TX, NUM_DELAY])).all() and (np.shape(fake_2) == np.array([NUM_FAKE_2, NUM_RX, NUM_TX, NUM_DELAY])).all():
    # K Nearest Calculating for 1
    sim_1, multi_1, multi_div_sim_1 = K_nearest(real_1_test, fake_1, NUM_RX, NUM_TX, NUM_DELAY, 1)
    # K Nearest Calculating for 2
    sim_2, multi_2, multi_div_sim_2 = K_nearest(real_2_test, fake_2, NUM_RX, NUM_TX, NUM_DELAY, 2)
    print('sim1 = ' + str(float(sim_1)) + ', multi1 = ' + str(float(multi_1)))
    print('sim2 = ' + str(float(sim_2)) + ', multi2 = ' + str(float(multi_2)))
    # Score 1 Calculating
    if sim_1 > 0.2 and multi_div_sim_1 < 20:
        score_1 = (20 - multi_div_sim_1) / 20
        print('Score1 = ' + str(float(score_1)))
    else:
        print('Score1 = 0.')
        score_1 = 0

    # Score 2 Calculating
    if sim_2 > 0.1 and multi_div_sim_2 < 40:
        score_2 = (40 - multi_div_sim_2) / 40
        print('Score2 = ' + str(float(score_2)))
    else:
        print('Score2 = 0.')
        score_2 = 0
    # Final Score Calculating
    score = (score_1 + score_2) / 2
    print('Score = ' + str(float(score)))
else:
    print('Invalid format.')
t4 = time.time()

print('Data loading: ' + str(t2-t1) +'s')
print('Data generating: ' + str(t3-t2) +'s')
print('Score calculating: ' + str(t4-t3) +'s')
print('Total time: ' + str(t4-t1) +'s')
#=======================================================================================================================
#=======================================================================================================================
