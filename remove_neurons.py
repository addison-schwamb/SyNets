import argparse
import json
import sys
import pickle
import numpy as np
load_dir = 'single_nets/'
save_dir = 'damaged_nets/'

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=json.loads)
args = parser.parse_args()
kwargs= args.d

def load_data(name,prefix,dir):
	filename = prefix + '_' + name
	with open(dir + filename, 'rb') as f:
		params, internal_x, x_ICs, r_ICs, error_ratio = pickle.load(f)

	return params, internal_x, x_ICs, r_ICs, error_ratio

def set_all_parameters(pct_rmv, name):
	params = dict()
	params['pct_rmv'] = pct_rmv
	params['name'] = name

	return params

def save_data_variable_size(*vars1, name=None, prefix='train', dir=None):
    file_name = prefix + '_' + name
    with open(dir + file_name, 'wb') as f:
        pickle.dump((vars1), f, protocol=-1)

init_params = set_all_parameters(**kwargs)
pct_rmv = init_params['pct_rmv']
params, internal_x, x_ICs, r_ICs, error_ratio = load_data(name=init_params['name'],prefix='train',dir=load_dir)
internal_x = internal_x[:,-1]

xlen = np.size(internal_x)
model_prs = params['model']
J = model_prs['J']
wi = model_prs['wi']
wo = model_prs['wo']
wd = model_prs['wd']
wf = model_prs['wf']
wfd = model_prs['wfd']

num_to_rmv = round(pct_rmv*xlen)
rmv_indices = np.random.randint(0,xlen,(num_to_rmv))

for i in range(num_to_rmv):
	internal_x = np.concatenate((internal_x[0:rmv_indices[i]],internal_x[rmv_indices[i]+1:]))
	J = np.concatenate((J[0:rmv_indices[i],:],J[rmv_indices[i]+1:,:]),axis=0)
	J = np.concatenate((J[:,0:rmv_indices[i]],J[:,rmv_indices[i]+1:]),axis=1)
	wi = np.concatenate((wi[0:rmv_indices[i],:],wi[rmv_indices[i]+1:,:]),axis=0)
	wo = np.concatenate((wo[0:rmv_indices[i],:],wo[rmv_indices[i]+1:,:]),axis=0)
	wd = np.concatenate((wd[0:rmv_indices[i],:],wd[rmv_indices[i]+1:,:]),axis=0)
	wf = np.concatenate((wf[0:rmv_indices[i],:],wf[rmv_indices[i]+1:,:]),axis=0)
	wfd = np.concatenate((wfd[0:rmv_indices[i],:],wfd[rmv_indices[i]+1:,:]),axis=0)

model_prs['J'] = J
model_prs['wi'] = wi
model_prs['wo'] = wo
model_prs['wd'] = wd
model_prs['wf'] = wf
model_prs['wfd'] = wfd
model_prs['N'] = np.size(internal_x)
params['model'] = model_prs

save_data_variable_size(params, internal_x, x_ICs, r_ICs, error_ratio, name=init_params['name'], prefix='damaged', dir=save_dir)
