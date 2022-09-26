import time
import numpy as np
import matplotlib.pyplot as plt

def initialize_net(params):
	net_prs = params['network']
	train_prs = params['train']
	msc_prs = params['msc']
	N = net_prs['N']
	sigma2 = net_prs['sigma2']
	rng = np.random.RandomState(msc_prs['seed'])

	Sigma = np.diagflat(sigma2 * np.ones([N, 1]))
	x = np.matmul(Sigma,rng.randn(N,1))
	J = 0.01 * rng.randn(N,N)
	wo = 0.2 * rng.rand(N, net_prs['d_output']) - 0.1
	wi = 0.2 * rng.rand(N, net_prs['d_input']) - 0.1

	return Sigma, x, J, wo, wi

def zero_fat_mats(params, is_train=True):
	net_prs = params['network']
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

	return x_mat, r_mat, eps_mat, u_mat, z_mat

def plot_matrix(mat):
	im = plt.imshow(mat)
	plt.colorbar(im)
	plt.show()

def train(params, dmg_params, dmg_x, exp_mat, input_digits):
	tic = time.time()

	dmg_model_prs = dmg_params['model']
	net_prs, dmg_net_prs = params['network'], dmg_params['network']
	train_prs = params['train']
	task_prs = params['task']
	msc_prs = params['msc']
	rng = np.random.RandomState(msc_prs['seed'])
	dt, tau = net_prs['dt'], dmg_net_prs['tau']
	alpha, eta = net_prs['alpha'], net_prs['eta']
	alphaR = net_prs['alphaR']
	N = net_prs['N']
	output = task_prs['output_encoding']
	train_steps = int((train_prs['n_train'] + train_prs['n_train_ext']) * task_prs['t_trial'] / net_prs['dt'])
	trial_steps = int((task_prs['t_trial']) / dt)
	Rbar = 0

	Sigma, x, J, wo, wi = initialize_net(params)
	r = np.tanh(x)
	u = np.matmul(wo.T, r)
	eps = np.matmul(Sigma,rng.randn(N,1))

	dmg_g, dmg_J, dmg_wi, dmg_wo = dmg_model_prs['g'], dmg_model_prs['J'], dmg_model_prs['wi'], dmg_model_prs['wo']
	dmg_wd, dmg_wf, dmg_wfd = dmg_model_prs['wd'], dmg_model_prs['wf'], dmg_model_prs['wfd']
	dmg_r = np.tanh(dmg_x)
	z = np.matmul(dmg_wo.T, dmg_r)
	zd = np.matmul(dmg_wd.T, dmg_r)

	x_mat, r_mat, eps_mat, u_mat, z_mat = zero_fat_mats(params, is_train=True)
	trial = 0
	i = 0
	for i in range(train_steps):
		x_mat[:, i] = x.reshape(-1)
		r_mat[:, i] = r.reshape(-1)
		eps_mat[:, i] = eps.reshape(-1)
		u_mat[i] = u
		z_mat[i] = z

		x = (1 - alpha)*x + alpha*(np.matmul(J, r) + np.matmul(wi, exp_mat[:, i].reshape([net_prs['d_input'],1]))) + eps
		r = np.tanh(x)
		u = np.matmul(wo.T, r)
		z = u
		#dmg_input = np.concatenate((u,exp_mat[:,i]),axis=None)
		#dmg_dx = -dmg_x + dmg_g * np.matmul(dmg_J, dmg_r) + np.matmul(dmg_wf, z) + np.matmul(dmg_wi, dmg_input.reshape([dmg_net_prs['d_input'],])) + np.matmul(dmg_wfd, zd)
		#dmg_x = dmg_x + (dmg_dx * dt) / tau
		#dmg_r = np.tanh(dmg_x)
		#z = np.matmul(dmg_wo.T, dmg_r)
		#zd = np.matmul(dmg_wd.T, dmg_r)

		if (i+1) % trial_steps == 0 and i != 0:
			eps_r = np.zeros([N, N, trial_steps])
			eps_in = np.zeros([N, 2, trial_steps])
			R = -abs(output[sum(input_digits[trial][1])] - z)

			for j in range(trial_steps):
				eps_r[:, :, j] = np.outer(eps_mat[:, int(i - trial_steps + j + 1)], r_mat[:, int(i - trial_steps + j)].T)
				eps_in[:, :, j] = np.outer(eps_mat[:, int(i - trial_steps + j + 1)], exp_mat[:, int(i - trial_steps + j)].T)

			#print('Input Digits: ', input_digits[trial])
			#print('z: ', np.around(2*z) / 2.0)
			#print(R-Rbar,'\n')

			deltaJ = eta * (R - Rbar) * np.sum(eps_r, axis=2)
			deltaWi = eta * (R - Rbar) * np.sum(eps_in, axis=2)
			Rbar = alphaR * Rbar + (1 - alphaR) * R

			J += deltaJ
			wi += deltaWi
			trial += 1

		eps = np.matmul(Sigma,rng.randn(N,1))

	toc = time.time()
	print('\n', 'train time = ', (toc-tic)/60)

	model_params = {'J': J, 'wo': wo, 'wi': wi}
	params['model'] = model_params
	task_prs['counter'] = i
	return x, dmg_x, params

def test(params, dmg_params, x_train, dmg_x, exp_mat, input_digits):
	model_prs, dmg_model_prs = params['model'], dmg_params['model']
	net_prs, dmg_net_prs = params['network'], dmg_params['network']
	train_prs = params['train']
	task_prs = params['task']
	msc_prs = params['msc']
	J, wo, wi = model_prs['J'], model_prs['wo'], model_prs['wi']
	dt, tau = net_prs['dt'], dmg_net_prs['tau']
	alpha = net_prs['alpha']
	dmg_g, dmg_J, dmg_wi, dmg_wo = dmg_model_prs['g'], dmg_model_prs['J'], dmg_model_prs['wi'], dmg_model_prs['wo']
	dmg_wd, dmg_wf, dmg_wfd = dmg_model_prs['wd'], dmg_model_prs['wf'], dmg_model_prs['wfd']
	test_steps = int(train_prs['n_test'] * task_prs['t_trial'] / net_prs['dt'])
	time_steps = np.arange(0, test_steps, 1)
	trial_steps = int((task_prs['t_trial']) / dt)
	counter = task_prs['counter']
	exp_mat = exp_mat[:, counter+1:]
	test_digits = input_digits[train_prs['n_train'] + train_prs['n_train_ext']:]

	i00, i01, i10, i11 = 0, 0, 0, 0

	x = x_train
	r = np.tanh(x)
	u = np.matmul(wo.T, r)
	dmg_r = np.tanh(dmg_x)
	z = np.matmul(dmg_wo.T, dmg_r)
	zd = np.matmul(dmg_wd.T, dmg_r)

	x_mat, r_mat, eps_mat, u_mat, z_mat = zero_fat_mats(params, is_train=False)
	trial = 0

	for i in range(test_steps):
		x_mat[:, i] = x.reshape(-1)
		r_mat[:, i] = r.reshape(-1)
		u_mat[i] = u
		z_mat[i] = z

		x = (1 - alpha)*x + alpha*(np.matmul(J, r) + np.matmul(wi, exp_mat[:, i].reshape([net_prs['d_input'],1])))
		r = np.tanh(x)
		u = np.matmul(wo.T, r)
		z = u
		#dmg_input = np.concatenate((u,exp_mat[:,i]),axis=None)
		#dmg_dx = -dmg_x + dmg_g * np.matmul(dmg_J, dmg_r) + np.matmul(dmg_wf, z) + np.matmul(dmg_wi, dmg_input.reshape([dmg_net_prs['d_input'],])) + np.matmul(dmg_wfd, zd)
		#dmg_x = dmg_x + (dmg_dx * dt) / tau
		#dmg_r = np.tanh(dmg_x)
		#z = np.matmul(dmg_wo.T, dmg_r)
		#zd = np.matmul(dmg_wd.T, dmg_r)

		if (i+1) % int((task_prs['t_trial'])/ dt ) == 0 and i != 0:


			if test_digits[trial][1] == (0,0) and i00 == 0:

				r00 = r_mat[:, i-1][:, np.newaxis]
				x00 = x_mat[:, i-1][:, np.newaxis]
				i00 = 1

			elif test_digits[trial][1] == (0,1) and i01 == 0:

				r01 = r_mat[:, i-1][:, np.newaxis]
				x01 = x_mat[:, i-1][:, np.newaxis]
				i01 = 1

			elif test_digits[trial][1] == (1,0) and i10 == 0:

				r10 = r_mat[:, i-1][:, np.newaxis]
				x10 = x_mat[:, i-1][:, np.newaxis]
				i10 = 1

			elif test_digits[trial][1] == (1,1) and i11 == 0:

				r11 = r_mat[:, i-1][:, np.newaxis]
				x11 = x_mat[:, i-1][:, np.newaxis]
				i11 = 1

			elif test_digits[trial][1] == (0, 2) and i02 == 0:

				r02 = r_mat[:, i - 1][:, np.newaxis]
				x02 = x_mat[:, i - 1][:, np.newaxis]
				i02 = 1

			elif test_digits[trial][1] == (2, 0) and i20 == 0:

				r20 = r_mat[:, i - 1][:, np.newaxis]
				x20 = x_mat[:, i - 1][:, np.newaxis]
				i20 = 1

			elif test_digits[trial][1] == (2, 2) and i22 == 0:

				r22 = r_mat[:, i - 1][:, np.newaxis]
				x22 = x_mat[:, i - 1][:, np.newaxis]
				i22 = 1

			elif test_digits[trial][1] == (1, 2) and i12 == 0:

				r12 = r_mat[:, i - 1][:, np.newaxis]
				x12 = x_mat[:, i - 1][:, np.newaxis]
				i12 = 1

			elif test_digits[trial][1] == (2, 1) and i21 == 0:

				r21 = r_mat[:, i - 1][:, np.newaxis]
				x21 = x_mat[:, i - 1][:, np.newaxis]
				i21 = 1

			print('Test Digits: ', test_digits[trial])
			print('z: ', np.around(2*z) / 2.0)
			trial += 1

	x_ICs = np.array([x00, x01, x10, x11])
	r_ICs = np.array([r00, r01, r10, r11])

	return x_ICs, r_ICs, x_mat, dmg_x
