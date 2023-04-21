import argparse
import json
import sys
import pickle
#from sklearn.decomposition import PCA
from SPM_task import *
from train_synets import *
from posthoc_tests import *
dir = ''

# parse arguments and set parameters for new network
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=json.loads)
args = parser.parse_args()
kwargs= args.d

def set_all_parameters(alpha, sigma2, max_grad, n_train, encoding, seed, damaged_net, feedback=True):
    params = dict()

    net_params = dict()
    net_params['N'] = 200
    net_params['dt'] = 0.1
    net_params['alpha'] = alpha
    net_params['sigma2'] = sigma2
    net_params['max_grad'] = max_grad
    net_params['alphaR'] = 0.3
    net_params['tau_phi'] = 300
    net_params['tau_p'] = 300
    if feedback:
        net_params['d_input'] = 4
    else:
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
    params['task'] = task_params

    train_params = dict()
    train_params['n_train'] = int(n_train[-1])  # training steps
    train_params['n_train_ext'] = int(0)
    train_params['n_test'] = int(20)      # test steps
    params['train'] = train_params

    other_params = dict()
    other_params['n_train'] = n_train
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
    try:
        with open(dir + filename, 'rb') as f:
            params, internal_x, x_ICs, r_ICs, error_ratio = pickle.load(f)
            return params, internal_x, x_ICs, r_ICs, error_ratio
    except:
        with open(dir + filename, 'rb') as f:
            params, internal_x, x_mat, err_mat, x_ICs, r_ICs, error_ratio = pickle.load(f)
            return params, internal_x, x_mat, err_mat, x_ICs, r_ICs, error_ratio


def save_data_variable_size(*vars1, name=None, prefix='train', dir=None):
    file_name = prefix + '_' + name
    with open(dir + file_name, 'wb') as f:
        pickle.dump((vars1), f, protocol=-1)

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


params = set_all_parameters(**kwargs)

# get MNIST digits and set up task
labels, digits_rep = get_digits_reps()
net_prs = params['network']
task_prs = params['task']
train_prs = params['train']
msc_prs = params['msc']
n_train = msc_prs['n_train']

task = sum_task_experiment(task_prs['n_digits'], train_prs['n_train'], train_prs['n_train_ext'], train_prs['n_test'], task_prs['time_intervals'],
                           net_prs['dt'], task_prs['output_encoding'], task_prs['keep_perms'] , digits_rep, labels, msc_prs['seed'])
exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()
train_steps = int((train_prs['n_train'] + train_prs['n_train_ext']) * task_prs['t_trial'] / net_prs['dt'])
trial_length = int(task_prs['t_trial'] / net_prs['dt'])

# load in damaged network
try:
    dmg_params, dmg_x, dmg_x_ICs, dmg_r_ICs, dmg_error_ratio = load_data(name=msc_prs['damaged_net'],prefix='damaged',dir=dir)
except:
    dmg_params, dmg_x, _, _, dmg_x_ICs, dmg_r_ICs, dmg_error_ratio = load_data(name=msc_prs['damaged_net'],prefix='damaged',dir=dir)


# test damaged network with no input
'''
filename = 'fp_test' + '_' + msc_prs['damaged_net']
with open(dir + filename, 'rb') as f:
    ph_params, _, _, _, attractor, _, _ = pickle.load(f)
print('Pre-Damage Attractor: ',attractor)


try:
    old_params, old_x, old_x_ICs, old_r_ICs, old_error_ratio = load_data(name=msc_prs['damaged_net'],prefix='train',dir=dir)
    old_x = old_x[:,-1]
except:
    old_params, old_x, _, _, old_x_ICs, old_r_ICs, old_error_ratio = load_data(name=msc_prs['damaged_net'],prefix='train',dir=dir)
old_params['model']['N'] = 1000
_, _, _, old_x_mat, old_err_mat = test_single(old_params, old_x, exp_mat, input_digits)
print('Average Error: ',np.mean(abs(old_err_mat)))
save_data_variable_size(old_params, old_x, old_x_mat, old_err_mat, old_x_ICs, old_r_ICs, old_error_ratio, name=msc_prs['damaged_net'],prefix='train',dir=dir)
'''

print('Post-Damage Performance')
dmg_params['network']['d_input'] = 2
dmg_x_ICs, dmg_r_ICs, dmg_x, dmg_x_mat, dmg_err_mat = test_single(dmg_params, dmg_x, exp_mat, input_digits)
print('Average Error: ',np.mean(abs(dmg_err_mat)))
R_prev = 0.25 - np.mean(abs(dmg_err_mat))
dmg_ph_params = set_posthoc_params(dmg_x_ICs, dmg_r_ICs)
save_data_variable_size(dmg_params, dmg_x, dmg_x_mat, dmg_err_mat, dmg_x_ICs, dmg_r_ICs, dmg_error_ratio, name=msc_prs['damaged_net'],prefix='damaged',dir=dir)

#trajectories, unique_z_mean, unique_zd_mean, attractor = attractor_type(dmg_params, dmg_ph_params, digits_rep, labels)
#print('Post-Damage Attractor: ',attractor)
#save_data_variable_size(ph_params, trajectories, unique_z_mean, unique_zd_mean, attractor, name=params['msc']['damaged_net'], prefix='damaged_fp_test', dir=dir)

# adjust params to allow for additional input
dmg_params = add_input_weights(dmg_params)

# train new network
train_steps = int((train_prs['n_train'] + train_prs['n_train_ext']) * task_prs['t_trial'] / net_prs['dt'])
params['task']['counter'] = train_steps
x_train, dmg_x, dmg_x_mat_trained, u_mat, params = train(params, dmg_params, dmg_x, exp_mat, target_mat, input_digits, R_prev)
x_ICs, r_ICs, internal_x, dmg_x_ICs, dmg_r_ICs, dmg_x, dmg_x_mat, u_mat, err_mat, _ = test(params, dmg_params, x_train, dmg_x, exp_mat, input_digits)
print('Average Error: ',np.mean(abs(err_mat)))
print('Energy (2-norm of u): ',np.linalg.norm(u_mat,2))

if msc_prs['feedback']:
    save_data_variable_size(params, internal_x, u_mat, err_mat, name=params['msc']['damaged_net']+'_fb', prefix='synet_trained', dir=dir)
else:
    save_data_variable_size(params, internal_x, u_mat, err_mat, name=params['msc']['damaged_net']+'_ff', prefix='synet_trained', dir=dir)

#ph_params = set_posthoc_params(x_ICs, r_ICs, dmg_x_ICs=dmg_x_ICs, dmg_r_ICs=dmg_r_ICs)


#trajectories, unique_z_mean, unique_zd_mean, attractor = attractor_type(params, ph_params, digits_rep, labels, synet=True, dmg_params=dmg_params)
#print('SyNet Attractor: ', attractor)
#save_data_variable_size(ph_params, trajectories, unique_z_mean, unique_zd_mean, attractor, name=params['msc']['damaged_net'], prefix='synet_fp_test', dir=dir)

#isHurwitz, kalmanRank, kalmanCond, synet_kalmanRank, synet_kalmanCond, input_inf = controllability_analysis(dmg_params)
#print('Hurwitz: ', isHurwitz)
#print('Kalman Matrix Rank: ', kalmanRank)
#print('Kalman Matrix Condition Number: ', kalmanCond)
#print('Kalman Matrix Rank (SyNet Input Only): ', synet_kalmanRank)
#print('Influence of SyNet Input: ', input_inf)
