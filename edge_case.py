import argparse
import json
import sys
import pickle
from SPM_task import *
from train_synets import *
from posthoc_tests import *
import matplotlib.pyplot as plt
dir = ''

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=json.loads)
args = parser.parse_args()
kwargs= args.d

def set_all_parameters(alpha, sigma2, max_grad, n_train, encoding, seed, damaged_net, feedback=True):
    params = dict()

    net_params = dict()
    net_params['N'] = 10
    net_params['dt'] = 0.1
    net_params['tau'] = 1
    net_params['alpha'] = alpha
    net_params['sigma2'] = sigma2
    net_params['max_grad'] = max_grad
    net_params['d_input'] = 2
    net_params['d_output'] = 1
    params['network'] = net_params

    task_params = dict()
    t_intervals = dict()
    t_intervals['fixate_on'], t_intervals['fixate_off'] = 0, 0
    t_intervals['cue_on'], t_intervals['cue_off'] = 0, 0
    t_intervals['stim_on'], t_intervals['stim_off'] = 10, 5
    t_intervals['delay_task'] = 0
    t_intervals['response'] = 5
    task_params['time_intervals'] = t_intervals
    task_params['t_trial'] = sum(t_intervals.values()) + t_intervals['stim_on'] + t_intervals['stim_off']
    task_params['output_encoding'] = encoding  # how 0, 1, 2 are encoded
    task_params['keep_perms'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    task_params['n_digits'] = 9
    task_params['counter'] = 0
    params['task'] = task_params

    train_params = dict()
    train_params['n_train'] = int(n_train)  # training steps
    train_params['n_train_ext'] = int(0)
    train_params['n_test'] = int(20)      # test steps
    params['train'] = train_params

    other_params = dict()
    other_params['seed'] = seed  #default is 0
    other_params['damaged_net'] = damaged_net
    other_params['feedback'] = feedback
    params['msc'] = other_params

    return params

def get_digits_reps():

    with open('allDigCNNMNIST', 'rb') as f:
        z_mean, z_log_var, z_sample = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    y_test, x_test = np.array(y_test), x_test.reshape([x_test.shape[0], 28, 28])

    return y_test, z_sample

def load_data(name,prefix,dir):
    filename = prefix + '_' + name
    with open(dir + filename, 'rb') as f:
        params, internal_x, x_ICs, r_ICs, error_ratio = pickle.load(f)

    return params, internal_x, x_ICs, r_ICs, error_ratio

def save_data_variable_size(*vars1, name=None, prefix='train', dir=None):
    file_name = prefix + '_' + name
    with open(dir + file_name, 'wb') as f:
        pickle.dump((vars1), f, protocol=-1)

def externalize_neurons(x, params):
    model_prs = params['model']
    J = model_prs['J']
    wi = model_prs['wi']
    wo = model_prs['wo']
    wd = model_prs['wd']
    wf = model_prs['wf']
    wfd = model_prs['wfd']

    dmg_x = x[0:990]
    ext_x = x[990:]
    dmg_J = J[0:990,0:990]
    ext_W = J[990:,990:]
    new_wi = J[0:990,990:]
    new_wd = J[990:,0:990]
    dmg_wi = wi[0:990,:]
    ext_wi = wi[990:,:]
    dmg_wo = wo[0:990]
    ext_wo = wo[990:]
    dmg_wd = wd[0:990,:]
    ext_wd = wd[990:,:]
    dmg_wf = wf[0:990]
    ext_wf = wf[990:]
    dmg_wfd = wfd[0:990,:]
    ext_wfd = wfd[990:,:]

    model_prs['J'] = dmg_J
    model_prs['wi'] = dmg_wi
    model_prs['wo'] = dmg_wo
    model_prs['wd'] = dmg_wd
    model_prs['new_wi'] = new_wi
    model_prs['new_wd'] = new_wd
    model_prs['wf'] = dmg_wf
    model_prs['wfd'] = dmg_wfd
    model_prs['N'] = np.size(dmg_x)
    params['model'] = model_prs

    ext_model_prs = dict()
    ext_model_prs['W'] = ext_W
    ext_model_prs['wi'] = ext_wi
    ext_model_prs['wd'] = ext_wd
    ext_model_prs['wo'] = ext_wo
    ext_model_prs['wf'] = ext_wf
    ext_model_prs['wfd'] = ext_wfd

    return ext_x, dmg_x, params, ext_model_prs

'''
def externalize_neurons(x, params, pct_rmv):
    xlen = np.size(x)
    model_prs = params['model']
    net_prs = params['network']
    J = model_prs['J']
    wi = model_prs['wi']
    wo = model_prs['wo']
    wd = model_prs['wd']
    wf = model_prs['wf']
    wfd = model_prs['wfd']
    d_input = net_prs['d_input']
    d_output = net_prs['d_output']

    num_to_rmv = round(pct_rmv*xlen)
    rmv_indices = np.random.randint(0,xlen,(num_to_rmv))
    ext_W = np.ones([num_to_rmv,num_to_rmv])

    dmg_x = np.concatenate((x[0:rmv_indices[0]], x[rmv_indices[0]+1:]))
    ext_x = x[rmv_indices[0]]
    ext_W[0, 0] = J[rmv_indices[0], rmv_indices[0]]
    new_wi = np.concatenate((J[rmv_indices[0],0:rmv_indices[0]], J[rmv_indices[0],rmv_indices[0]+1:]), axis=0).reshape((1,xlen-1))
    new_wd = np.concatenate((J[0:rmv_indices[0],rmv_indices[0]], J[rmv_indices[0]+1:,rmv_indices[0]])).reshape((xlen-1,1))
    J = np.concatenate((J[0:rmv_indices[0],:], J[rmv_indices[0]+1:,:]), axis=0)
    J = np.concatenate((J[:,0:rmv_indices[0]], J[:,rmv_indices[0]+1:]), axis=1)
    ext_wi = wi[rmv_indices[0],:].reshape((1,d_input))
    wi = np.concatenate((wi[0:rmv_indices[0],:], wi[rmv_indices[0]+1:,:]), axis=0)
    ext_wo = wo[rmv_indices[0],:].reshape((1,d_output))
    wo = np.concatenate((wo[0:rmv_indices[0],:], wo[rmv_indices[0]+1:,:]), axis=0)
    ext_wd = wd[rmv_indices[0],:].reshape((1,d_input))
    wd = np.concatenate((wd[0:rmv_indices[0],:], wd[rmv_indices[0]+1:,:]), axis=0)
    ext_wf = wf[rmv_indices[0],:].reshape((1,d_output))
    wf = np.concatenate((wf[0:rmv_indices[0],:], wf[rmv_indices[0]+1:,:]), axis=0)
    ext_wfd = wfd[rmv_indices[0],:].reshape((1,d_input))
    wfd = np.concatenate((wfd[0:rmv_indices[0],:], wfd[rmv_indices[0]+1:,:]), axis=0)

    for i in range(1,num_to_rmv):
        dmg_x = np.concatenate((dmg_x[0:rmv_indices[i]], dmg_x[rmv_indices[i]+1:]))
        ext_x = np.append(ext_x, x[rmv_indices[i]])
        ext_W[i, 0:i] = new_wi[:, rmv_indices[i]]
        ext_W[0:i, i] = new_wd[rmv_indices[i], :]
        ext_W[i, i] = J[rmv_indices[i], rmv_indices[i]]
        new_wi = np.append(new_wi, J[rmv_indices[i],:].reshape((1,xlen-i)), axis=0)
        new_wi = np.concatenate((new_wi[:,0:rmv_indices[i]], new_wi[:,rmv_indices[i]+1:]), axis=1)
        new_wd = np.append(new_wd, J[:,rmv_indices[i]].reshape((xlen-i,1)), axis=1)
        new_wd = np.concatenate((new_wd[0:rmv_indices[i],:], new_wd[rmv_indices[i]+1:,:]), axis=0)
        J = np.concatenate((J[0:rmv_indices[i],:], J[rmv_indices[i]+1:,:]), axis=0)
        J = np.concatenate((J[:,0:rmv_indices[i]], J[:,rmv_indices[i]+1:]), axis=1)
        ext_wi = np.append(ext_wi, wi[rmv_indices[i],:].reshape((1,d_input)), axis=0)
        ext_wi = np.concatenate((ext_wi[:,0:rmv_indices[i]], ext_wi[:,rmv_indices[i]+1:]), axis=1)
        wi = np.concatenate((wi[0:rmv_indices[i],:], wi[rmv_indices[i]+1:,:]), axis=0)
        ext_wo = np.append(ext_wo, wo[rmv_indices[i],:].reshape((1,d_output)), axis=0)
        ext_wo = np.concatenate((ext_wo[:,0:rmv_indices[i]], ext_wo[:,rmv_indices[i]+1:]), axis=1)
        wo = np.concatenate((wo[0:rmv_indices[i],:], wo[rmv_indices[i]+1:,:]), axis=0)
        ext_wd = np.append(ext_wd, wd[rmv_indices[i],:].reshape((1,d_input)), axis=0)
        ext_wd = np.concatenate((ext_wd[:,0:rmv_indices[i]], ext_wd[:,rmv_indices[i]+1:]), axis=1)
        wd = np.concatenate((wd[0:rmv_indices[i],:], wd[rmv_indices[i]+1:,:]), axis=0)
        ext_wf = np.append(ext_wf, wf[rmv_indices[i],:].reshape((1,d_output)), axis=0)
        ext_wf = np.concatenate((ext_wf[:,0:rmv_indices[i]], ext_wf[:,rmv_indices[i]+1:]), axis=1)
        wf = np.concatenate((wf[0:rmv_indices[i],:], wf[rmv_indices[i]+1:,:]), axis=0)
        ext_wfd = np.append(ext_wfd, wfd[rmv_indices[i],:].reshape((1,d_input)), axis=0)
        ext_wfd = np.concatenate((ext_wfd[:,0:rmv_indices[i]], ext_wfd[:,rmv_indices[i]+1:]), axis=1)
        wfd = np.concatenate((wfd[0:rmv_indices[i],:], wfd[rmv_indices[i]+1:,:]), axis=0)


    model_prs['J'] = J
    model_prs['wi'] = wi
    model_prs['wo'] = wo
    model_prs['wd'] = wd
    model_prs['new_wi'] = new_wi
    model_prs['new_wd'] = new_wd
    model_prs['wf'] = wf
    model_prs['wfd'] = wfd
    model_prs['N'] = np.size(dmg_x)
    params['model'] = model_prs

    ext_model_prs = dict()
    ext_model_prs['W'] = ext_W
    ext_model_prs['wi'] = ext_wi
    ext_model_prs['wd'] = ext_wd
    ext_model_prs['wo'] = ext_wo
    ext_model_prs['wf'] = ext_wf
    ext_model_prs['wfd'] = ext_wfd

    return ext_x, dmg_x, params, ext_model_prs
'''

def train_edge(ext_params, dmg_params, ext_x, dmg_x, exp_mat, target_mat, input_digits):
    tic = time.time()
    ext_model_prs, dmg_model_prs = ext_params['model'], dmg_params['model']
    ext_net_prs, dmg_net_prs = ext_params['network'], dmg_params['network']
    train_prs = ext_params['train']
    task_prs = ext_params['task']
    msc_prs = ext_params['msc']
    rng = np.random.RandomState(msc_prs['seed'])
    d_input, d_output = ext_net_prs['d_input'], ext_net_prs['d_output']
    dt, tau = ext_net_prs['dt'], ext_net_prs['tau']
    alpha, sigma2, max_grad = ext_net_prs['alpha'], ext_net_prs['sigma2'], ext_net_prs['max_grad']
    ext_wi, ext_wd, ext_wo = ext_model_prs['wi'], ext_model_prs['wd'], ext_model_prs['wo']
    ext_wf, ext_wfd = ext_model_prs['wf'], ext_model_prs['wfd']
    N, output = ext_net_prs['N'], task_prs['output_encoding']
    train_steps = int((train_prs['n_train'] + train_prs['n_train_ext']) * task_prs['t_trial'] / dt)
    trial_steps = int((task_prs['t_trial']) / dt)

    Sigma = np.diagflat(ext_net_prs['sigma2'] * np.ones([N, 1]))
    #ext_x = np.matmul(Sigma,rng.randn(N,1))
    W = 0.01 * rng.randn(N,N)
    ext_r = np.tanh(ext_x)
    ext_z = np.matmul(ext_wo.T, ext_r)
    ext_zd = np.matmul(ext_wd.T, ext_r)
    eps = np.matmul(Sigma,rng.randn(N,1))

    dmg_g, dmg_J, dmg_wi, dmg_wo = dmg_model_prs['g'], dmg_model_prs['J'], dmg_model_prs['wi'], dmg_model_prs['wo']
    dmg_wd, dmg_wf, dmg_wfd = dmg_model_prs['wd'], dmg_model_prs['wf'], dmg_model_prs['wfd']
    new_wi, new_wd = dmg_model_prs['new_wi'], dmg_model_prs['new_wd']
    dmg_r = np.tanh(dmg_x)
    dmg_z = np.matmul(dmg_wo.T, dmg_r)
    dmg_zd = np.matmul(dmg_wd.T, dmg_r)

    x_mat, r_mat, eps_mat, ext_z_mat, z_mat, zd_mat, rwd_mat, deltaW_mat = zero_fat_mats(ext_params, is_train=True)
    dmg_x_mat = np.zeros([np.shape(dmg_x)[0], train_steps])
    trial = 0
    i = 0
    last_stop = 0
    for i in range(train_steps):
        x_mat[:, i] = ext_x.reshape(-1)
        dmg_x_mat[:, i] = dmg_x.reshape(-1)
        r_mat[:, i] = ext_r.reshape(-1)
        eps_mat[:, i] = eps.reshape(-1)
        ext_z_mat[i] = ext_z
        z_mat[i] = dmg_z
        zd_mat[:, i] = dmg_zd.reshape(-1)

        ext_dx = -ext_x + dmg_g*(np.matmul(W, ext_r)) + np.matmul(ext_wf, dmg_z) + np.matmul(ext_wfd, ext_zd+dmg_zd) + np.matmul(ext_wi, exp_mat[:,i]) + eps.reshape(-1)
        dmg_dx = -dmg_x + dmg_g*(np.matmul(dmg_J, dmg_r) + np.matmul(new_wi, ext_r)) + np.matmul(dmg_wf, dmg_z) + np.matmul(dmg_wfd, ext_zd+dmg_zd) + np.matmul(dmg_wi, exp_mat[:,i])
        ext_x = ext_x + (ext_dx * dt) / tau
        dmg_x = dmg_x + (dmg_dx * dt) / tau
        ext_r = np.tanh(ext_x)
        ext_z = np.matmul(ext_wo.T, ext_r)
        ext_zd = np.matmul(ext_wd.T, ext_r)
        dmg_r = np.tanh(dmg_x)
        dmg_z = np.matmul(dmg_wo.T, dmg_r)
        dmg_zd = np.matmul(dmg_wd.T, dmg_r)

        if np.any(target_mat[:, i] != 0.):
            eps_r = np.zeros([N, N])
            R = 0.25 - abs(target_mat[:,i] - (ext_z + dmg_z))
            rwd_mat[trial] = R
            steps_since_update = i - last_stop

            for j in range(1, steps_since_update+1):
                idx = int(i - steps_since_update + j)
                r_cum = 0
                for k in range(1, j+1):
                    r_cum += ((1 - dt) ** (k - 1)) * dt * r_mat[:, idx-k]
                eps_r += np.outer(eps_mat[:, idx], r_cum)

            deltaW = (alpha * dt / sigma2) * R * eps_r
            if np.linalg.norm(deltaW, ord='fro') > max_grad:
                deltaW = (max_grad / np.linalg.norm(deltaW, ord='fro')) * deltaW
            #deltaW_mat[trial] = deltaW

            W += deltaW
            last_stop = i

        if (i+1) % trial_steps == 0 and i != 0:
            trial += 1

        eps = np.matmul(Sigma, rng.randn(N, 1))

    toc = time.time()
    print('\n', 'train time = ', (toc-tic)/60)

    ext_model_prs['W'] = W
    ext_model_prs['Sigma'] = Sigma
    ext_params['model'] = ext_model_prs
    task_prs['counter'] = i

    plt.figure()
    plt.plot(rwd_mat)
    plt.show()
    return ext_x, dmg_x, dmg_x_mat, ext_params

def test_edge(ext_params, dmg_params, ext_x, dmg_x, exp_mat, input_digits):
    ext_model_prs, dmg_model_prs = ext_params['model'], dmg_params['model']
    ext_net_prs, dmg_net_prs = ext_params['network'], dmg_params['network']
    train_prs = ext_params['train']
    task_prs = ext_params['task']
    msc_prs = ext_params['msc']

    d_input, d_output = ext_net_prs['d_input'], ext_net_prs['d_output']
    dt, tau = ext_net_prs['dt'], ext_net_prs['tau']
    alpha, N, output = ext_net_prs['alpha'], ext_net_prs['N'], task_prs['output_encoding']
    W, ext_wi, ext_wd, ext_wo = ext_model_prs['W'], ext_model_prs['wi'], ext_model_prs['wd'], ext_model_prs['wo']
    ext_wf, ext_wfd = ext_model_prs['wf'], ext_model_prs['wfd']
    dmg_g, dmg_J, dmg_wi, dmg_wo = dmg_model_prs['g'], dmg_model_prs['J'], dmg_model_prs['wi'], dmg_model_prs['wo']
    dmg_wd, dmg_wf, dmg_wfd = dmg_model_prs['wd'], dmg_model_prs['wf'], dmg_model_prs['wfd']
    new_wi, new_wd = dmg_model_prs['new_wi'], dmg_model_prs['new_wd']
    test_steps = int(train_prs['n_test'] * task_prs['t_trial'] / dt)
    time_steps = np.arange(0, test_steps, 1)
    trial_steps = int((task_prs['t_trial']) / dt)
    counter = task_prs['counter']
    exp_mat = exp_mat[:, counter+1:]
    test_digits = input_digits[train_prs['n_train'] + train_prs['n_train_ext']:]

    i00, i01, i10, i11 = 0, 0, 0, 0

    ext_r = np.tanh(ext_x)
    ext_z = np.matmul(ext_wo.T, ext_r)
    ext_zd = np.matmul(ext_wd.T, ext_r)
    dmg_r = np.tanh(dmg_x)
    dmg_z = np.matmul(dmg_wo.T, dmg_r)
    dmg_zd = np.matmul(dmg_wd.T, dmg_r)

    x_mat, r_mat, eps_mat, ext_z_mat, z_mat, zd_mat, rwd_mat, deltaW_mat = zero_fat_mats(ext_params, is_train=False)
    dmg_x_mat = np.zeros([np.shape(dmg_x)[0], test_steps])
    dmg_r_mat = np.zeros([np.shape(dmg_x)[0], test_steps])
    trial = 0

    for i in range(test_steps):
        x_mat[:, i] = ext_x.reshape(-1)
        dmg_x_mat[:, i] = dmg_x.reshape(-1)
        r_mat[:, i] = ext_r.reshape(-1)
        dmg_r_mat[:, i] = dmg_r.reshape(-1)
        ext_z_mat[i] = ext_z
        z_mat[i] = dmg_z
        zd_mat[:, i] = dmg_zd.reshape(-1)

        ext_dx = -ext_x + dmg_g*(np.matmul(W, ext_r)) + np.matmul(ext_wf, dmg_z) + np.matmul(ext_wfd, ext_zd+dmg_zd) + np.matmul(ext_wi, exp_mat[:,i])
        dmg_dx = -dmg_x + dmg_g*(np.matmul(dmg_J, dmg_r) + np.matmul(new_wi, ext_r)) + np.matmul(dmg_wf, dmg_z) + np.matmul(dmg_wfd, ext_zd+dmg_zd) + np.matmul(dmg_wi, exp_mat[:,i])
        ext_x = ext_x + (ext_dx * dt) / tau
        dmg_x = dmg_x + (dmg_dx * dt) / tau
        ext_r = np.tanh(ext_x)
        ext_z = np.matmul(ext_wo.T, ext_r)
        ext_zd = np.matmul(ext_wd.T, ext_r)
        dmg_r = np.tanh(dmg_x)
        dmg_z = np.matmul(dmg_wo.T, dmg_r)
        dmg_zd = np.matmul(dmg_wd.T, dmg_r)

        if (i+1) % int((task_prs['t_trial'])/ dt ) == 0 and i != 0:


            if test_digits[trial][1] == (0,0) and i00 == 0:

                r00 = r_mat[:, i-1][:, np.newaxis]
                x00 = x_mat[:, i-1][:, np.newaxis]
                dmg_r00 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x00 = dmg_x_mat[:, i-1][:, np.newaxis]
                i00 = 1

            elif test_digits[trial][1] == (0,1) and i01 == 0:

                r01 = r_mat[:, i-1][:, np.newaxis]
                x01 = x_mat[:, i-1][:, np.newaxis]
                dmg_r01 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x01 = dmg_x_mat[:, i-1][:, np.newaxis]
                i01 = 1

            elif test_digits[trial][1] == (1,0) and i10 == 0:

                r10 = r_mat[:, i-1][:, np.newaxis]
                x10 = x_mat[:, i-1][:, np.newaxis]
                dmg_r10 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x10 = dmg_x_mat[:, i-1][:, np.newaxis]
                i10 = 1

            elif test_digits[trial][1] == (1,1) and i11 == 0:

                r11 = r_mat[:, i-1][:, np.newaxis]
                x11 = x_mat[:, i-1][:, np.newaxis]
                dmg_r11 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x11 = dmg_x_mat[:, i-1][:, np.newaxis]
                i11 = 1

            elif test_digits[trial][1] == (0, 2) and i02 == 0:

                r02 = r_mat[:, i - 1][:, np.newaxis]
                x02 = x_mat[:, i - 1][:, np.newaxis]
                dmg_r02 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x02 = dmg_x_mat[:, i-1][:, np.newaxis]
                i02 = 1

            elif test_digits[trial][1] == (2, 0) and i20 == 0:

                r20 = r_mat[:, i - 1][:, np.newaxis]
                x20 = x_mat[:, i - 1][:, np.newaxis]
                dmg_r20 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x20 = dmg_x_mat[:, i-1][:, np.newaxis]
                i20 = 1

            elif test_digits[trial][1] == (2, 2) and i22 == 0:

                r22 = r_mat[:, i - 1][:, np.newaxis]
                x22 = x_mat[:, i - 1][:, np.newaxis]
                dmg_r22 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x22 = dmg_x_mat[:, i-1][:, np.newaxis]
                i22 = 1

            elif test_digits[trial][1] == (1, 2) and i12 == 0:

                r12 = r_mat[:, i - 1][:, np.newaxis]
                x12 = x_mat[:, i - 1][:, np.newaxis]
                dmg_r12 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x12 = dmg_x_mat[:, i-1][:, np.newaxis]
                i12 = 1

            elif test_digits[trial][1] == (2, 1) and i21 == 0:

                r21 = r_mat[:, i - 1][:, np.newaxis]
                x21 = x_mat[:, i - 1][:, np.newaxis]
                dmg_r21 = dmg_r_mat[:, i-1][:, np.newaxis]
                dmg_x21 = dmg_x_mat[:, i-1][:, np.newaxis]
                i21 = 1

            print('Test Digits: ', test_digits[trial])
            print('z: ', np.around(2*(dmg_z)) / 2.0)
            trial += 1

    x_ICs = np.array([x00, x01, x10, x11])
    r_ICs = np.array([r00, r01, r10, r11])
    dmg_x_ICs = np.array([dmg_x00, dmg_x01, dmg_x10, dmg_x11])
    dmg_r_ICs = np.array([dmg_r00, dmg_r01, dmg_r10, dmg_r11])

    return x_ICs, r_ICs, x_mat, dmg_x_ICs, dmg_r_ICs, dmg_x


ext_params = set_all_parameters(**kwargs)

# get MNIST digits and set up task
labels, digits_rep = get_digits_reps()
ext_net_prs = ext_params['network']
ext_task_prs = ext_params['task']
ext_train_prs = ext_params['train']
ext_msc_prs = ext_params['msc']

task = sum_task_experiment(ext_task_prs['n_digits'], ext_train_prs['n_train'], ext_train_prs['n_train_ext'], ext_train_prs['n_test'], ext_task_prs['time_intervals'],
                           ext_net_prs['dt'], ext_task_prs['output_encoding'], ext_task_prs['keep_perms'], digits_rep, labels, ext_msc_prs['seed'])
exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()

old_params, old_x, _, _, _ = load_data(name=ext_msc_prs['damaged_net'],prefix='train',dir=dir)
old_x = old_x[:,-1]
old_params['model']['N'] = 1000
old_x_ICs, old_r_ICs, _, old_x_mat = test_single(old_params, old_x, exp_mat, input_digits)

print('\n')
ext_x, dmg_x, dmg_params, ext_params['model'] = externalize_neurons(old_x, old_params)
dmg_x_ICs, dmg_r_ICs, _, dmg_x_mat = test_single(dmg_params, dmg_x, exp_mat, input_digits)

print('\n')
ext_x, dmg_x, dmg_x_mat, ext_params = train_edge(ext_params, dmg_params, ext_x, dmg_x, exp_mat, target_mat, input_digits)
x_ICs, r_ICs, ext_x, dmg_x_ICs, dmg_r_ICs, dmg_x = test_edge(ext_params, dmg_params, ext_x, dmg_x, exp_mat, input_digits)
ph_params = set_posthoc_params(x_ICs, r_ICs, dmg_x_ICs=dmg_x_ICs, dmg_r_ICs=dmg_r_ICs)
