import argparse
import json
import sys
import pickle
from SPM_task import *
from train_synets import *
dir = ''

# parse arguments and set parameters for new network
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=json.loads)
args = parser.parse_args()
kwargs= args.d

def set_all_parameters(alpha, sigma2, n_train, encoding, seed, damaged_net):
	params = dict()

	net_params = dict()
	net_params['N'] = 1000
	net_params['dt'] = 0.1
	net_params['alpha'] = alpha
	#net_params['alphaR'] = alphaR
	net_params['sigma2'] = sigma2
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
	train_params['n_train'] = int(n_train)  # training steps
	train_params['n_train_ext'] = int(0)
	train_params['n_test'] = int(20)	  # test steps
	params['train'] = train_params

	other_params = dict()
	other_params['seed'] = seed	 #default is 0
	other_params['damaged_net'] = damaged_net
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

task = sum_task_experiment(task_prs['n_digits'], train_prs['n_train'], train_prs['n_train_ext'], train_prs['n_test'], task_prs['time_intervals'],
						   net_prs['dt'], task_prs['output_encoding'], task_prs['keep_perms'] , digits_rep, labels, msc_prs['seed'])
exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()

# load in damaged network
dmg_params, dmg_x, _, _, _ = load_data(name=msc_prs['damaged_net'],prefix='damaged',dir=dir)

# test damaged network with no input
dmg_x = test_single(dmg_params, dmg_x, exp_mat, input_digits)

# adjust params to allow for additional input
dmg_params = add_input_weights(dmg_params)

# train new network
x_train, dmg_x, params = train(params, dmg_params, dmg_x, exp_mat, target_mat, input_digits)
x_ICs, r_ICs, internal_x, dmg_x = test(params, dmg_params, x_train, dmg_x, exp_mat, input_digits)
