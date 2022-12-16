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
    W = 0.01 * rng.randn(N,N)
    wo = 0.2 * rng.rand(N, net_prs['d_output']) - 0.1
    wi = 0.2 * rng.rand(N, net_prs['d_input']) - 0.1

    return Sigma, x, W, wo, wi

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
    zd_mat = np.zeros([2, total_steps])
    rwd_mat = np.zeros(total_size)
    deltaW_mat = np.zeros(total_size)

    return x_mat, r_mat, eps_mat, u_mat, z_mat, zd_mat, rwd_mat, deltaW_mat

def plot_matrix(mat):
    im = plt.imshow(mat)
    plt.colorbar(im)
    plt.show()

def train(params, dmg_params, dmg_x, exp_mat, target_mat, input_digits):
    tic = time.time()

    dmg_model_prs = dmg_params['model']
    net_prs, dmg_net_prs = params['network'], dmg_params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    rng = np.random.RandomState(msc_prs['seed'])
    dt, tau = net_prs['dt'], dmg_net_prs['tau']
    alpha, sigma2, max_grad = net_prs['alpha'], net_prs['sigma2'], net_prs['max_grad']
    #alphaR = net_prs['alphaR']
    N = net_prs['N']
    output = task_prs['output_encoding']
    train_steps = int((train_prs['n_train'] + train_prs['n_train_ext']) * task_prs['t_trial'] / net_prs['dt'])
    trial_steps = int((task_prs['t_trial']) / dt)
    Rbar = 0

    Sigma, x, W, wo, wi = initialize_net(params)
    r = np.tanh(x)
    u = np.matmul(wo.T, r)
    eps = np.matmul(Sigma,rng.randn(N,1))

    dmg_g, dmg_J, dmg_wi, dmg_wo = dmg_model_prs['g'], dmg_model_prs['J'], dmg_model_prs['wi'], dmg_model_prs['wo']
    dmg_wd, dmg_wf, dmg_wfd = dmg_model_prs['wd'], dmg_model_prs['wf'], dmg_model_prs['wfd']
    dmg_r = np.tanh(dmg_x)
    z = np.matmul(dmg_wo.T, dmg_r)
    zd = np.matmul(dmg_wd.T, dmg_r)

    x_mat, r_mat, eps_mat, u_mat, z_mat, zd_mat, rwd_mat, deltaW_mat = zero_fat_mats(params, is_train=True)
    dmg_x_mat = np.zeros([np.shape(dmg_x)[0], train_steps])
    trial = 0
    i = 0
    changes = 0
    last_stop = 0
    for i in range(train_steps):
        x_mat[:, i] = x.reshape(-1)
        dmg_x_mat[:, i] = dmg_x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)
        eps_mat[:, i] = eps.reshape(-1)
        u_mat[i] = u
        z_mat[i] = z
        zd_mat[:, i] = zd.reshape(-1)

        input = np.concatenate((zd,exp_mat[:,i]), axis=None)
        x = (1 - dt)*x + dt*(np.matmul(W, r) + np.matmul(wi, input.reshape([net_prs['d_input'],1]))) + eps
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
            R = 0.25 - abs(target_mat[:,i] - z)
            rwd_mat[trial] = R
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

            #print('Input Digits: ', input_digits[trial])
            #print('z: ', np.around(2*z) / 2.0)
            #print('R: ', R)

            deltaW =  (alpha * dt / sigma2) * R * eps_r
            if np.linalg.norm(deltaW,ord='fro') > max_grad:
                deltaW = (max_grad/np.linalg.norm(deltaW,ord='fro'))*deltaW
            deltawi = (alpha * dt / sigma2) * R * eps_in
            deltaW_mat[trial] = np.linalg.norm(deltaW,ord='fro')

            W += deltaW
            wi += deltawi
            last_stop = i

        if (i+1) % trial_steps == 0 and i != 0:
            trial += 1

        eps = np.matmul(Sigma,rng.randn(N,1))

    toc = time.time()
    print('\n', 'train time = ', (toc-tic)/60)

    model_params = {'W': W, 'wo': wo, 'wi': wi, 'Sigma': Sigma}
    params['model'] = model_params
    task_prs['counter'] = i

    #plt.plot(rwd_mat)
    #plt.figure()
    #plt.plot(deltaW_mat)
    #plt.show()

    return x, dmg_x, dmg_x_mat, params

def test(params, dmg_params, x_train, dmg_x, exp_mat, input_digits):
    model_prs, dmg_model_prs = params['model'], dmg_params['model']
    net_prs, dmg_net_prs = params['network'], dmg_params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    rng = np.random.RandomState(msc_prs['seed'])
    W, wo, wi = model_prs['W'], model_prs['wo'], model_prs['wi']
    dt, tau = net_prs['dt'], dmg_net_prs['tau']
    alpha = net_prs['alpha']
    #Sigma, N = model_prs['Sigma'], net_prs['N']
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
    #eps = np.matmul(Sigma,rng.randn(N,1))
    dmg_r = np.tanh(dmg_x)
    z = np.matmul(dmg_wo.T, dmg_r)
    zd = np.matmul(dmg_wd.T, dmg_r)

    dmg_x_mat, dmg_r_mat, eps_mat, u_mat, z_mat, zd_mat, rwd_mat, deltaW_mat = zero_fat_mats(dmg_params, is_train=False)
    x_mat = np.zeros([net_prs['N'],np.shape(dmg_x_mat)[1]])
    r_mat = np.zeros([net_prs['N'],np.shape(dmg_x_mat)[1]])
    trial = 0

    for i in range(test_steps):
        dmg_x_mat[:, i] = dmg_x.reshape(-1)
        dmg_r_mat[:, i] = dmg_r.reshape(-1)
        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)
        u_mat[i] = u
        z_mat[i] = z
        zd_mat[:, i] = zd.reshape(-1)

        input = np.concatenate((zd,exp_mat[:,i]), axis=None)
        x = (1 - dt)*x + dt*(np.matmul(W, r) + np.matmul(wi, input.reshape([net_prs['d_input'],1]))) #+ eps
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

            print('Test Digits: ', test_digits[trial])
            print('z: ', np.around(2*z) / 2.0)
            trial += 1

    x_ICs = np.array([x00, x01, x10, x11])
    r_ICs = np.array([r00, r01, r10, r11])
    dmg_x_ICs = np.array([dmg_x00, dmg_x01, dmg_x10, dmg_x11])
    dmg_r_ICs = np.array([dmg_r00, dmg_r01, dmg_r10, dmg_r11])

    return x_ICs, r_ICs, x_mat, dmg_x_ICs, dmg_r_ICs, dmg_x

def test_single(params, x, exp_mat, input_digits):
    net_prs = params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    model_prs = params['model']
    params['network']['N'] = params['model']['N']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    test_steps = int(train_prs['n_test'] * task_prs['t_trial'] / net_prs['dt'])
    time_steps = np.arange(0, test_steps, 1)

    i00, i01, i10, i11 = 0, 0, 0, 0

    r = np.tanh(x)
    z = np.matmul(wo.T, r)
    zd = np.matmul(wd.T, r)

    x_mat, r_mat, eps_mat, u_mat, z_mat, zd_mat, rwd_mat, deltaW_mat = zero_fat_mats(params, is_train=False)
    trial = 0

    for i in range(test_steps):
        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)
        z_mat[i] = z
        zd_mat[:, i] = zd.reshape(-1)

        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, exp_mat[:,i].reshape([net_prs['d_input'],])) + np.matmul(wfd, zd)
        x = x + (dx * dt) / tau
        r = np.tanh(x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)

        if (i+1) % int((task_prs['t_trial']) / dt) == 0:
            if input_digits[trial][1] == (0,0) and i00 == 0:

                r00 = r_mat[:, i-1][:, np.newaxis]
                x00 = x_mat[:, i-1][:, np.newaxis]
                i00 = 1

            elif input_digits[trial][1] == (0,1) and i01 == 0:

                r01 = r_mat[:, i-1][:, np.newaxis]
                x01 = x_mat[:, i-1][:, np.newaxis]
                i01 = 1

            elif input_digits[trial][1] == (1,0) and i10 == 0:

                r10 = r_mat[:, i-1][:, np.newaxis]
                x10 = x_mat[:, i-1][:, np.newaxis]
                i10 = 1

            elif input_digits[trial][1] == (1,1) and i11 == 0:

                r11 = r_mat[:, i-1][:, np.newaxis]
                x11 = x_mat[:, i-1][:, np.newaxis]
                i11 = 1

            elif input_digits[trial][1] == (0, 2) and i02 == 0:

                r02 = r_mat[:, i - 1][:, np.newaxis]
                x02 = x_mat[:, i - 1][:, np.newaxis]
                i02 = 1

            elif input_digits[trial][1] == (2, 0) and i20 == 0:

                r20 = r_mat[:, i - 1][:, np.newaxis]
                x20 = x_mat[:, i - 1][:, np.newaxis]
                i20 = 1

            elif input_digits[trial][1] == (2, 2) and i22 == 0:

                r22 = r_mat[:, i - 1][:, np.newaxis]
                x22 = x_mat[:, i - 1][:, np.newaxis]
                i22 = 1

            elif input_digits[trial][1] == (1, 2) and i12 == 0:

                r12 = r_mat[:, i - 1][:, np.newaxis]
                x12 = x_mat[:, i - 1][:, np.newaxis]
                i12 = 1

            elif input_digits[trial][1] == (2, 1) and i21 == 0:

                r21 = r_mat[:, i - 1][:, np.newaxis]
                x21 = x_mat[:, i - 1][:, np.newaxis]
                i21 = 1
            print('Test Digits: ', input_digits[trial])
            print('z: ', np.around(2*z) / 2.0)
            trial += 1

    x_ICs = np.array([x00, x01, x10, x11])
    r_ICs = np.array([r00, r01, r10, r11])

    return x_ICs, r_ICs, x, x_mat
