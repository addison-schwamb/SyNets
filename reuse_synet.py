import argparse
import json
import sys
import pickle
from SPM_task import *
from train_synets import *
from posthoc_tests import *
dir = ''

# parse arguments and set parameters for new network
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=json.loads)
args = parser.parse_args()
kwargs= args.d

def set_all_parameters(n_train, encoding, seed, damaged_net, synet):
    params = dict()

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
    params['task'] = task_params

    train_params = dict()
    train_params['n_train'] = int(n_train[-1])  # training steps
    train_params['n_train_ext'] = int(0)
    train_params['n_test'] = int(20)      # test steps
    params['train'] = train_params

    other_params = dict()
    other_params['seed'] = seed  #default is 0
    other_params['damaged_net'] = damaged_net
    other_params['synet'] = synet
    other_params['n_train'] = n_train
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
        if prefix == 'damaged':
            params, internal_x, _, _, _, _, _ = pickle.load(f)
        elif prefix == 'synet_trained':
            params, internal_x, _, _ = pickle.load(f)

    return params, internal_x

def add_input_weights(params):
    rng = np.random.RandomState(msc_prs['seed'])
    net_prs = params['network']
    train_prs = params['train']
    model_prs = params['model']
    wi = model_prs['wi']
    N = model_prs['N']

    if train_prs['init_dist'] == 'Gauss':
        new_weights = (1. * rng.randn(N, 1)) / net_prs['input_var']
    elif train_prs['init_dist'] == 'Uniform':
        new_weights = (2 * rng.rand(N, 1) - 1) / net_prs['input_var']

    new_wi = np.concatenate((new_weights,wi),axis=1)
    model_prs['wi'] = new_wi
    net_prs['d_input'] += 1
    net_prs['N'] = model_prs['N']
    params['model'] = model_prs
    params['network'] = net_prs

    return params

def zero_fat_mats_reuse(params, net_prs, is_train=True):
    train_prs = params['train']
    task_prs = params['task']

    if is_train:
        total_size = train_prs['n_train'] + train_prs['n_train_ext']
    elif not is_train:
        total_size = train_prs['n_test']

    total_steps = int(total_size * task_prs['t_trial'] / net_prs['dt'])
    x_mat = np.zeros([net_prs['N'], total_steps])
    r_mat = np.zeros([net_prs['N'], total_steps])
    eps_mat = np.zeros([net_prs['N'], total_steps])
    u_mat = np.zeros(total_steps)
    z_mat = np.zeros(total_steps)
    zd_mat = np.zeros([2, total_steps])
    rwd_mat = np.zeros(50*total_size)
    deltaW_mat = np.zeros(total_size)

    return x_mat, r_mat, eps_mat, u_mat, z_mat, zd_mat, rwd_mat, deltaW_mat

def train_reuse(params, synet_params, dmg_params, synet_x, dmg_x, exp_mat, target_mat, input_digits):
    tic = time.time()

    model_prs, dmg_model_prs = synet_params['model'], dmg_params['model']
    net_prs, dmg_net_prs = synet_params['network'], dmg_params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    rng = np.random.RandomState(msc_prs['seed'])
    dt, tau = net_prs['dt'], dmg_net_prs['tau']
    alpha, sigma2, max_grad = net_prs['alpha'], net_prs['sigma2'], net_prs['max_grad']
    alphaR = net_prs['alphaR']
    N = net_prs['N']
    output = task_prs['output_encoding']
    train_steps = int((train_prs['n_train'] + train_prs['n_train_ext']) * task_prs['t_trial'] / net_prs['dt'])
    trial_steps = int((task_prs['t_trial']) / dt)
    Rbar = 0
    n_correct = 0;

    Sigma, W, wo, wi = model_prs['Sigma'], model_prs['W'], model_prs['wo'], model_prs['wi']
    x = synet_x
    r = np.tanh(x)
    u = np.matmul(wo.T, r)
    b = np.zeros([net_prs['N'],])
    eps = np.matmul(Sigma,rng.randn(N,1))

    dmg_g, dmg_J, dmg_wi, dmg_wo = dmg_model_prs['g'], dmg_model_prs['J'], dmg_model_prs['wi'], dmg_model_prs['wo']
    dmg_wd, dmg_wf, dmg_wfd = dmg_model_prs['wd'], dmg_model_prs['wf'], dmg_model_prs['wfd']
    dmg_wi = dmg_wi[:,-3:]
    dmg_r = np.tanh(dmg_x)
    z = np.matmul(dmg_wo.T, dmg_r)
    zd = np.matmul(dmg_wd.T, dmg_r)

    x_mat, r_mat, eps_mat, u_mat, z_mat, zd_mat, rwd_mat, deltaW_mat = zero_fat_mats_reuse(params, net_prs, is_train=True)
    dmg_x_mat = np.zeros([np.shape(dmg_x)[0], train_steps])
    trial = 0
    i = 0
    rwd_ind = 0
    last_stop = 0
    prev_correct = 0
    R_prev = 0
    for i in range(train_steps):
        x_mat[:, i] = x.reshape(-1)
        dmg_x_mat[:, i] = dmg_x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)
        eps_mat[:, i] = eps.reshape(-1)
        u_mat[i] = u
        z_mat[i] = z
        zd_mat[:, i] = zd.reshape(-1)

        input = np.concatenate((zd,exp_mat[:,i]), axis=None)
        x = (1 - dt)*x + dt*(np.matmul(W, r) + np.matmul(wi, input.reshape([net_prs['d_input'],]))) + eps.reshape([N,])
        r = np.tanh(x)
        u = np.matmul(wo.T, r)
        dmg_input = np.concatenate((u,exp_mat[:,i]),axis=None)
        dmg_dx = -dmg_x + dmg_g * np.matmul(dmg_J, dmg_r) + np.matmul(dmg_wf, z) + np.matmul(dmg_wi, dmg_input.reshape([dmg_net_prs['d_input'],])) + np.matmul(dmg_wfd, zd)
        dmg_x = dmg_x + (dmg_dx * dt) / tau
        dmg_r = np.tanh(dmg_x)
        z = np.matmul(dmg_wo.T, dmg_r)
        zd = np.matmul(dmg_wd.T, dmg_r)

        #if (i+1) % trial_steps == 0 and i != 0:
        if np.any(target_mat[:, i] != 0.):
            #print(target_mat[:,i])
            eps_r = np.zeros([N, N])
            eps_in = np.zeros([N, net_prs['d_input']])
            eps_cum = np.zeros([N,1])
            R = 0.25 - abs(target_mat[:,i] - z)
            rwd_mat[rwd_ind] = R
            Rbar = alphaR*Rbar + (1 - alphaR)*R
            rwd_ind += 1
            #Rbar = np.mean(rwd_mat[:trial+1])
            steps_since_update = i - last_stop

            for j in range(1, steps_since_update+1):
                idx = int(i - steps_since_update + j)
                r_cum = 0
                input_cum = 0
                for k in range(1, j+1):
                    r_cum += ((1 - dt) ** (k - 1)) * dt * r_mat[:, idx-k]
                    input_cum += ((1 - dt) ** (k - 1)) * dt * np.concatenate((zd_mat[:, idx-k], exp_mat[:, idx-k]), axis=None)
                eps_r += np.outer(eps_mat[:, idx], r_cum)
                eps_in += np.outer(eps_mat[:, idx], input_cum)
                eps_cum += eps_mat[:,idx].reshape([N,1])

            #print('Input Digits: ', input_digits[trial])
            #print('z: ', np.around(2*z) / 2.0)
            #print('R: ', R)

            deltaW =  (alpha * dt / sigma2) * (R) * eps_r
            if np.linalg.norm(deltaW,ord='fro') > max_grad:
                deltaW = (max_grad/np.linalg.norm(deltaW,ord='fro'))*deltaW
            deltawi = (alpha * dt / sigma2) * (R) * eps_in
            deltab = (alpha * dt / sigma2) * (R)* eps_cum
            deltaW_mat[trial] = np.linalg.norm(deltaW,ord='fro')

            W += deltaW
            wi += deltawi
            b += deltab.reshape([N,])
            phi_s = (1 - 1/tau_phi)*phi_s + (1/tau_phi)*R
            phi_l = (1 - 1/tau_phi)*phi_l + (1/tau_phi)*phi_s
            phi = phi_s - phi_l;
            R_prev = R
            last_stop = i

        if (i+1) % trial_steps == 0 and i != 0:
            if trial in msc_prs['n_train']:
                model_params = {'W': W, 'wo': wo, 'wi': wi, 'Sigma': Sigma, 'b': b}
                synet_params['model'] = model_params
                print('\n','n = ',trial)
                _, _, _, _, _, _, _, _, _, n_correct = test_reuse(params, synet_params, dmg_params, x, dmg_x, exp_mat, input_digits)
                print('Correct: ',n_correct)
                #sigma2 = ((20 - n_correct - prev_correct)/20)*sigma2 + 0.001
                #alphaR = ((20 - n_correct - prev_correct)/20)*alphaR
                prev_correct = n_correct

            trial += 1

        Sigma = np.diagflat(sigma2 * np.ones([N, 1]))
        eps = np.matmul(Sigma,rng.randn(N,1))

    toc = time.time()
    print('\n', 'train time = ', (toc-tic)/60)

    model_params = {'W': W, 'wo': wo, 'wi': wi, 'Sigma': Sigma, 'b': b}
    synet_params['model'] = model_params
    task_prs['counter'] = i

    #plt.plot(rwd_mat)
    #plt.figure()
    #plt.plot(deltaW_mat)
    #plt.show()

    return x, dmg_x, dmg_x_mat, synet_params

def test_reuse(params, synet_params, dmg_params, x_train, dmg_x, exp_mat, input_digits):
    model_prs, dmg_model_prs = synet_params['model'], dmg_params['model']
    net_prs, dmg_net_prs = synet_params['network'], dmg_params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    rng = np.random.RandomState(msc_prs['seed'])
    W, wo, wi, b = model_prs['W'], model_prs['wo'], model_prs['wi'], model_prs['b']
    dt, tau = net_prs['dt'], dmg_net_prs['tau']
    alpha = net_prs['alpha']
    Sigma, N = model_prs['Sigma'], net_prs['N']
    dmg_g, dmg_J, dmg_wi, dmg_wo = dmg_model_prs['g'], dmg_model_prs['J'], dmg_model_prs['wi'], dmg_model_prs['wo']
    dmg_wd, dmg_wf, dmg_wfd = dmg_model_prs['wd'], dmg_model_prs['wf'], dmg_model_prs['wfd']
    dmg_wi = dmg_wi[:,-3:]
    test_steps = int(train_prs['n_test'] * task_prs['t_trial'] / net_prs['dt'])
    time_steps = np.arange(0, test_steps, 1)
    trial_steps = int((task_prs['t_trial']) / dt)
    counter = task_prs['counter']
    exp_mat = exp_mat[:, counter+1:]
    test_digits = input_digits[train_prs['n_train'] + train_prs['n_train_ext']:]
    encoding = task_prs['output_encoding']

    i00, i01, i10, i11 = 0, 0, 0, 0

    x = x_train
    r = np.tanh(x)
    u = np.matmul(wo.T, r)
    eps = np.matmul(Sigma,rng.randn(N,1))
    dmg_r = np.tanh(dmg_x)
    z = np.matmul(dmg_wo.T, dmg_r)
    zd = np.matmul(dmg_wd.T, dmg_r)

    dmg_x_mat, dmg_r_mat, eps_mat, u_mat, z_mat, zd_mat, rwd_mat, deltaW_mat = zero_fat_mats(dmg_params, is_train=False)
    x_mat = np.zeros([net_prs['N'],np.shape(dmg_x_mat)[1]])
    r_mat = np.zeros([net_prs['N'],np.shape(dmg_x_mat)[1]])
    err_mat = np.zeros([train_prs['n_test'],])
    trial = 0
    n_correct = 0

    for i in range(test_steps-1):
        dmg_x_mat[:, i] = dmg_x.reshape(-1)
        dmg_r_mat[:, i] = dmg_r.reshape(-1)
        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)
        u_mat[i] = u
        z_mat[i] = z
        zd_mat[:, i] = zd.reshape(-1)

        input = np.concatenate((zd,exp_mat[:,i]), axis=None)
        x = (1 - dt)*x + dt*(np.matmul(W, r) + np.matmul(wi, input.reshape([net_prs['d_input'],]))) + eps.reshape([N,])
        r = np.tanh(x)
        u = np.matmul(wo.T, r)
        dmg_input = np.concatenate((u,exp_mat[:,i]),axis=None)
        dmg_dx = -dmg_x + dmg_g * np.matmul(dmg_J, dmg_r) + np.matmul(dmg_wf, z) + np.matmul(dmg_wi, dmg_input.reshape([dmg_net_prs['d_input'],])) + np.matmul(dmg_wfd, zd)
        dmg_x = dmg_x + (dmg_dx * dt) / tau
        dmg_r = np.tanh(dmg_x)
        z = np.matmul(dmg_wo.T, dmg_r)
        zd = np.matmul(dmg_wd.T, dmg_r)
        #eps = np.matmul(Sigma,rng.randn(N,1))

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

            err_mat[trial] = encoding[sum(test_digits[trial][1])] - z
            if np.around(2*z)/2.0 == encoding[sum(test_digits[trial][1])]:
                n_correct += 1
            print('Test Digits: ', test_digits[trial])
            print('z: ', np.around(2*z) / 2.0)
            trial += 1

    x_ICs = np.array([x00, x01, x10, x11])
    r_ICs = np.array([r00, r01, r10, r11])
    dmg_x_ICs = np.array([dmg_x00, dmg_x01, dmg_x10, dmg_x11])
    dmg_r_ICs = np.array([dmg_r00, dmg_r01, dmg_r10, dmg_r11])

    return x_ICs, r_ICs, x_mat, dmg_x_ICs, dmg_r_ICs, dmg_x, dmg_x_mat, u_mat, err_mat, n_correct

params = set_all_parameters(**kwargs)
task_prs = params['task']
train_prs = params['train']
msc_prs = params['msc']
labels, digits_rep = get_digits_reps()
dmg_params, dmg_x = load_data(name=msc_prs['damaged_net'],prefix='damaged',dir=dir)
dmg_params = add_input_weights(dmg_params)
synet_params, synet_x = load_data(name=msc_prs['synet'], prefix='synet_trained', dir=dir)

train_steps = int((train_prs['n_train'] + train_prs['n_train_ext']) * task_prs['t_trial'] / synet_params['network']['dt'])
params['task']['counter'] = train_steps

task = sum_task_experiment(task_prs['n_digits'], train_prs['n_train'], train_prs['n_train_ext'], train_prs['n_test'], task_prs['time_intervals'],
    synet_params['network']['dt'], task_prs['output_encoding'], task_prs['keep_perms'] , digits_rep, labels, msc_prs['seed'])
exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()

print('Training with Pre-trained SyNet')
x_train, dmg_x, dmg_x_mat_trained, synet_params = train_reuse(params, synet_params, dmg_params, synet_x[:,-1], dmg_x, exp_mat, target_mat, input_digits)
x_ICs, r_ICs, internal_x, dmg_x_ICs, dmg_r_ICs, dmg_x, dmg_x_mat, u_mat, err_mat, n_correct = test_reuse(params, synet_params, dmg_params, x_train, dmg_x, exp_mat, input_digits)
print('Correct: ',n_correct)
params['network'] = synet_params['network']
params['model'] = synet_params['model']
params['msc']['feedback'] = True
#ph_params = set_posthoc_params(x_ICs, r_ICs, dmg_x_ICs=dmg_x_ICs, dmg_r_ICs=dmg_r_ICs)
#trajectories, unique_z_mean, unique_zd_mean, attractor = attractor_type(params, ph_params, digits_rep, labels, synet=True, dmg_params=dmg_params)
#print('SyNet Attractor: ', attractor)
