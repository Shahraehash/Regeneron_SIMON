import os 
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from utils.model_utils import *
from model.SSSDS4Imputer import SSSDS4Imputer
import json
import torch
import torch.nn as nn
import tensorflow as tf

def fast5_to_npy(fast5_folder):
    #Save reads in list and measure longest read length
    fast5_data = []
    longest_read_length = -1 * float("inf")    
    for fast5_file in os.listdir(fast5_folder):
        if fast5_file.endswith('.fast5'):
            with get_fast5_file(os.path.join(fast5_folder, fast5_file), mode = "r") as f5:
                for read in f5.get_reads():
                    fast5_data += [read.get_raw_data()]
                    if len(read.get_raw_data()) > longest_read_length:
                        longest_read_length = len(read.get_raw_data())
    
    #Convert reads in list to numpy array with same shape by padding zeros
    fast5_npy_array = np.zeros((len(fast5_data), 1, longest_read_length))
    for i in range(len(fast5_data)):
        for j in range(len(fast5_data[i])):
            fast5_npy_array[i, 0, j] = fast5_data[i][j]
    
    return fast5_npy_array, np.mean(fast5_npy_array, axis= 0)[0], np.std(fast5_npy_array, axis = 0)[0], longest_read_length

def hyperparameter_tuning(T_range, beta0_range, betaT_range, train_config, trainset_config, diffusion_config, wavenet_config):
    ideal_T, ideal_beta0, ideal_betaT = 0, 0, 0
    
    #Hyperparameter training of T
    curr_avg_best = float("inf")
    curr_best = float("inf")
    for T in T_range:
        diffusion_config["T"] = T
        vals = training(train_config, trainset_config, diffusion_config, wavenet_config)
        loss_vals = [float(value[1]) for value in vals]
        if loss_vals[0] > loss_vals[-1] and sum(loss_vals)/len(loss_vals) < curr_avg_best and abs(loss_vals[-1] - curr_best) > 0.0001:
            print('updated T')
            ideal_T = T
            curr_avg_best = sum(loss_vals)/len(loss_vals)
            curr_best = loss_vals[-1]
    diffusion_config["T"] = ideal_T
    
    #Hyperparameter training of Beta0
    curr_avg_best = float("inf")
    curr_best = float("inf")
    for beta0 in beta0_range:
        diffusion_config["beta_0"] = beta0 
        vals = training(train_config, trainset_config, diffusion_config, wavenet_config)
        loss_vals = [float(value[1]) for value in vals]
        if loss_vals[0] > loss_vals[-1] and sum(loss_vals)/len(loss_vals) < curr_avg_best and abs(loss_vals[-1] - curr_best) > 0.0001:
            print('updated beta0')
            ideal_beta0 = beta0
            curr_avg_best = sum(loss_vals)/len(loss_vals)
            curr_best = loss_vals[-1]
    diffusion_config["beta_0"] = ideal_beta0
    
    #Hyperparameter training of BetaT
    curr_avg_best = float("inf")
    curr_best = float("inf")
    for betaT in betaT_range:
        diffusion_config["beta_T"] = betaT
        vals = training(train_config, trainset_config, diffusion_config, wavenet_config)
        loss_vals = [float(value[1]) for value in vals]
        if loss_vals[0] > loss_vals[-1] and sum(loss_vals)/len(loss_vals) < curr_avg_best and abs(loss_vals[-1] - curr_best) > 0.0001:
            print('updated betaT')
            ideal_betaT = betaT
            curr_avg_best = sum(loss_vals)/len(loss_vals)
            curr_best = loss_vals[-1]
    diffusion_config["beta_T"] = ideal_betaT        

    return ideal_T, ideal_beta0, ideal_betaT


def train(output_directory, ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging, learning_rate, only_generate_missing, missing_k, mean, stdev):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    
    
    local_path = "T{}_beta0{}_betaT{}_rm".format(diffusion_config["T"], diffusion_config["beta_0"], diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSDS4Imputer(**model_config).cuda()
    print_size(net)
    training_loss_vals = []
    
    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
        
    if ckpt_iter >= 0:
        try:
            print('loading_checkpoint')
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
              
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')


    if os.path.exists(output_directory + "/training_loss.txt"): 
        with open(output_directory + "/training_loss.txt", "r") as f:
            lines = f.readlines()
        for elem in lines:
            values = elem.split(",")
            training_loss_vals += [(int(values[0][1:]), float(values[1][:-3]))]
            
    
    ### Custom data loading and reshaping ###
    training_data = np.load(trainset_config['train_data_path'])
    training_data = np.split(training_data, training_data.shape[0]/5, 0)
    training_data = np.array(training_data)
    training_data = torch.from_numpy(training_data).float().cuda()
    print('Data loaded')

    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch in training_data:
            
            transposed_mask = get_mask_rm(batch[0], missing_k)
            
            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams, mean, stdev, only_generate_missing=only_generate_missing)

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))
                training_loss_vals += [(n_iter, loss.item())]
            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1
    
    if training_loss_vals != []:
        with open(output_directory + "/training_loss.txt", "w+") as f:
            for loss in training_loss_vals:
                f.write(str(loss) + "\n")
        f.close()
        
    return training_loss_vals

def training(train_config, training_config, diff_config, wavenet_config):

    global trainset_config
    trainset_config = training_config  # to load trainset

    global diffusion_config
    diffusion_config = diff_config # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = wavenet_config

    training_loss_vals = train(**train_config)
    
    return training_loss_vals