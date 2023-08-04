import os
import random
import numpy as np
import time
from math import log10, floor, log
from dtaidistance import dtw
from fastdtw import fastdtw
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
import scipy.stats as st
from statistics import mean
#from ont_fast5_api.fast5_file import Fast5File
from pod5.tools.pod5_convert_to_fast5 import convert_to_fast5
from pathlib import Path
import shutil

import matplotlib.pyplot as plt

from model.SSSDS4Imputer import *
from utils.model_utils import *
from utils.training_utils import fast5_to_npy
from utils.evaluation_utils import fastdtw_process
from utils.creating_fast5 import *

### Load Kmer model mapping 6-mer to (min_val, max_val) of signal for 6-mer ###
def load_kmer_model():
    with open(os.getcwd() + "/utils/kmer_database.txt") as f:
        lines = f.readlines()
    #creat dictionary where key is the 6-mer and value is a tuple (min, max) of interval based on mean and stdev
    kmer_dict = dict()
    header = True
    for line in lines:
        split_line = line.split('\t')
        if header:
            header = False
            continue
        kmer_dict[split_line[0]] = [float(split_line[1]) - float(split_line[2]), float(split_line[1]) + float(split_line[2])]
    
    return kmer_dict

def rep_rvs(size,a=0.1, more=1, seed=0):
    a = a*5
    array_1 = np.ones(int(size*(0.075-0.015*a))).astype(int)
    #Distribution of read lengths
    samples = st.alpha.rvs(3.3928495261646932+a,
        -7.6451557771999035+(2*a), 50.873948369526737,
        size=(size-int(size*(0.075-0.015*a))), random_state=seed).astype(int)
    samples = np.concatenate((samples, array_1), 0)
    samples[samples<0] = 0
    samples[samples>40] = 40
    if more == 1:
        np.random.seed(seed)
        addi = np.array(abs(np.random.normal(2,1,size))).astype(int)
        samples[samples<9] += addi[samples<9]
        np.random.shuffle(samples)
        samples[samples<9] += addi[samples<9]
    return samples

def repeat_n_time(a, result, more, seed=0):
    rep_times = rep_rvs(len(result), a, more, seed)
    out = list()
    ali = list()
    pos = 0
    for i in range(len(result)):
        k = rep_times[i]
        cur = [result[i]] * k
        out.extend(cur)
        for j in range(k):
            ali.append((pos,i))
            pos = pos + 1
    event_idx = np.repeat(np.arange(len(result)), rep_times)
    return out,ali,event_idx

def decode_sequence(sequence, kmer_dict):
    random.seed(13)
    #Create Ground Truth Signal
    signal_data = []
    for i in range(len(sequence) - 5):
        signal_data += [random.uniform(kmer_dict[sequence[i:i+6]][0], kmer_dict[sequence[i:i+6]][1])]
    
    #Adjust length based on distribution of read lengths
    expect_signal, final_ali, event_idx = repeat_n_time(0.1, signal_data, 1, seed=0)
    shift = np.median(expect_signal)
    scale = np.median(np.abs(expect_signal - shift))
    return expect_signal


def fasta_to_npy(fasta_file, max_len_sequence):
    kmer_dict = load_kmer_model()
    #Decode the sequence from kmer model and save signal data in list
    with open(fasta_file, "r") as fasta:
        lines = fasta.readlines()
    fasta.close()
    fasta_data = []
    for idx in range(int(len(lines)/2)):
        if len(decode_sequence(lines[2*idx + 1][:-1], kmer_dict)) <= max_len_sequence:
            fasta_data += [[int(elem) for elem in decode_sequence(lines[2*idx + 1][:-1], kmer_dict)]]
    
    #Convert reads in list to numpy array with same shape by padding zeros
    npy_fasta_array = np.zeros((len(fasta_data), 1, max_len_sequence))
    for i in range(len(fasta_data)):
        for j in range(min(len(fasta_data[i]), max_len_sequence)):
            npy_fasta_array[i,0,j] = fasta_data[i][j]
    
    return npy_fasta_array, np.nanmean(npy_fasta_array, axis = 0)[0], np.nanstd(npy_fasta_array, axis = 0)[0], max_len_sequence

def trim_zeros(data):
    last = len(data)
    for idx, val in enumerate(data[:last][::-1]):
        if (val < 1 and val >= 0) or (val > -1 and val <= 0):
            last -= 1
        else:
           break
    return data[:last]

def add_gaussian_noise(data):
    return np.random.normal(np.array(data), scale = 1).tolist()

def post_processing(data_list):
    new_data_list = []
    for data in data_list:
        new_data_list += [[int(val) for val in add_gaussian_noise(trim_zeros(data))]]
    return new_data_list

def save_fast5(data_list, path):
    #Read slow5 file
    header, metadata = read_slow5("/home/raehash.shah/SIMON/model/example.slow5")
    
    temp_data_list = data_list
    index = 0
    while len(temp_data_list) > 0:
        #Perform post-processing
        curr_data_list = post_processing(temp_data_list[:4000])
        #Write slow5
        write_slow5(header, metadata, curr_data_list, path + f"slow5_{index}.slow5")
        #convert it to fast5
        convert_slow5_to_fast5(path + f"slow5_{index}.slow5", path + f"fast5_{index}.fast5")
        temp_data_list = temp_data_list[4000:]
        index += 1
    
    return


### Generate Simulated Data ###
def generate(output_directory, num_samples, max_sample_len, ckpt_path, data_path, ckpt_iter, missing_k, only_generate_missing):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """
    
    

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}_rm".format(diffusion_config["T"], diffusion_config["beta_0"], diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

            
    # predefine model
    net = SSSDS4Imputer(**model_config).cuda()
    print_size(net)

    
    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

        
        
    ### Custom data loading and reshaping ###  
    testing_data = np.load(data_path)
    testing_points = testing_data.shape[0]
    testing_data = np.split(testing_data, 1, 0) # So that all of the .fasta are a part of the sample
    testing_data = np.array(testing_data)
    testing_data = torch.from_numpy(testing_data).float().cuda()
    
    print('Data loaded')


    all_mse = []
    for i, batch in enumerate(testing_data):
        bigger_generated_audio = []
        bigger_batch = []
        bigger_mask = []
        for iteration in range(int(math.floor(num_samples/testing_points))):
            batch_copy = batch.clone()
            batch_copy = torch.nan_to_num(batch_copy, nan = 0.0)
            print("Generating " + str(iteration*testing_points) + " - " + str((iteration + 1)*testing_points) + " points")
            mask_T = get_mask_rm(batch_copy[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch_copy.size()[0], 1, 1).float().cuda()

                
                
            batch_copy = batch_copy.permute(0,2,1)
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            sample_length = batch_copy.size(2)
            sample_channels = batch_copy.size(1)
            generated_audio = sampling(net, (testing_points, sample_channels, sample_length), diffusion_hyperparams, train_config["mean"], train_config["stdev"], cond=batch_copy, mask=mask, only_generate_missing=only_generate_missing)

            end.record()
            torch.cuda.synchronize()

            print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(testing_points,
                                                                                                ckpt_iter,
                                                                                                int(start.elapsed_time(
                                                                                                    end) / 1000)))

            
            
            generated_audio = generated_audio.detach().cpu().numpy()
            batch_copy = batch_copy.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            
            
            bigger_generated_audio += generated_audio.tolist()
            bigger_batch += batch_copy.tolist()
            bigger_mask += mask.tolist()
        
        generated_audio = np.array(bigger_generated_audio)
        batch = np.array(bigger_batch)
        mask = np.array(bigger_mask)
        
        
        
        outfile = f'all_generated_data_{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, np.transpose(batch, (0,2,1)))

        print('saved generated samples at iteration %s' % ckpt_iter)
        mse = compute_mse(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)], max_sample_len)
        all_mse.append(mse)
    
    return all_mse, new_out

def normalized(output_array, original_array):
    print(np.min(output_array), np.max(output_array), np.min(original_array), np.max(original_array))
    return ((output_array - np.min(output_array)) * (1/(np.max(output_array) - np.min(output_array)))) * (np.max(original_array) - np.min(original_array)) + np.min(original_array) 


def compute_mse(npy_array1, npy_array2, max_sample_len):
    squared_val = []
    for idx, val in np.ndenumerate(npy_array1):
        if idx[0] > max_sample_len:
            break
        squared_val += [(npy_array1[idx] - npy_array2[idx]) ** 2]
    return sum(squared_val)


def generate_sample(num_samples, max_sample_len, diffusion, wavenet, train, testing_data, gen):

    # Parse configs. Globals nicer in this case
    gen_config = gen

    global train_config
    train_config = train # training parameters

    global diffusion_config
    diffusion_config = diffusion  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = wavenet

    list_mse = generate(**gen_config, ckpt_iter=train_config["ckpt_iter"], num_samples=num_samples, max_sample_len = max_sample_len, data_path=testing_data, missing_k=train_config["missing_k"], only_generate_missing=train_config["only_generate_missing"])

    return list_mse
