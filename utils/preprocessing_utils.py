import os
import numpy as np
import random
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.fast5_file import Fast5File
from pathlib import Path
from fastdtw import fastdtw
from scipy.spatial import distance
from pod5.tools.pod5_convert_to_fast5 import convert_to_fast5
from itertools import product

from utils.creating_fast5 import *


### Mapping readid: [seq, sig] ####
def map_readid_sequence(fasta_file):
    read_id_to_sequence_dict = dict()
    with open(fasta_file, "r") as fasta:
        lines = fasta.readlines()
    fasta.close()
    for idx in range(int(len(lines)/2)):
        read_id_to_sequence_dict[lines[2*idx]] = lines[2*idx + 1]
    return read_id_to_sequence_dict

def map_readid_sequence_signal(fast5_folder, read_id_dict):
    # Create map of read_id: [sequence, signal]
    read_id_to_seq_sig = dict()
    for fast5_file in os.listdir(fast5_folder):
        if fast5_file.endswith(".fast5"):
            with get_fast5_file(os.path.join(fast5_folder, fast5_file), mode = "r") as f5:
                for read in f5.get_reads():
                    if read.get_read_id() in read_id_dict:
                        read_id_to_seq_sig[read.get_read_id()] = [read_id_dict[read.get_read_id()], read.get_raw_data().tolist()]
    return read_id_to_seq_sig

#############################################################

### Generating Database ###

def load_kmer_model():
    with open(os.getcwd() + "/utils/ONT_kmer_model.txt") as f: #Use ONT to start the mapping
        lines = f.readlines()
    #creat dictionary where key is the 6-mer and value is just the mean
    kmer_dict = dict()
    header = True
    for line in lines:
        split_line = line.split('\t')
        if header:
            header = False
            continue
        kmer_dict[split_line[0]] = float(split_line[1])
    
    return kmer_dict

def ONT_signal(sequence, ONT_dict):
    sequence_data = []
    signal_data = []
    for i in range(len(sequence)-5):
        sequence_data += [sequence[i:i+6]]
        signal_data += [ONT_dict[sequence[i:i+6]]]
    return sequence_data, signal_data


def create_database(fast5_fasta_mapping):
    random.seed(28)
    #Create kmer_signal_database.txt where each 6-mer: avg(signal), stdev(signal)
    kmer_dict = dict()
    all_kmers = [''.join(b) for b in product('ATCG', repeat=6)]
    for kmer in all_kmers:
        kmer_dict[kmer] = []
    
    ONT_kmer_model = load_kmer_model()
    
    random_reads = random.sample(list(fast5_fasta_mapping.values()), len(list(fast5_fasta_mapping.values())))
    for index, (fast5, fasta) in enumerate(random_reads):
        print(f"Mapping the {index} value")
        #Create DTW mapping of a decoded signal value to the true signal value and then work backwards to map sequence to true signal value
        spliced_sequence, ONT_signal_data = ONT_signal(fasta, ONT_kmer_model)
        _, path = fastdtw_process(ONT_signal_data, fast5)
        for (x,y) in path:
            kmer_dict[spliced_sequence[x]] += [fast5[y]]
    
    kmer_signal_metrics = compute_signal_metrics(kmer_dict)
    
    kmer_file = open(os.getcwd()+"/utils/kmer_database.txt", "w+")
    kmer_file.write("kmer \t level_mean \t level_stdv \n")
    for kmer in kmer_signal_metrics:
        kmer_file.write(kmer + "\t" + str(kmer_signal_metrics[kmer][0]) + "\t" + str(kmer_signal_metrics[kmer][1]) + "\n")
    
    kmer_file.close()
    return



def fastdtw_process(read1, read2):
    
    read1_x = [i for i in range(len(read1))]
    read2_x = [j for j in range(len(read2))]
    
    read1_new = np.hstack((np.array(read1_x).reshape(len(read1_x), 1), np.array(read1).reshape(len(read1), 1)))
    read2_new = np.hstack((np.array(read2_x).reshape(len(read2_x), 1), np.array(read2).reshape(len(read2), 1)))
    
    dtw_distance, warp_path = fastdtw(read1_new, read2_new, dist = distance.minkowski)
    
    return dtw_distance, warp_path 

def compute_signal_metrics(kmer_dict):
    new_kmer_dict = dict()
    for kmer in kmer_dict:
        kmer_val_list = np.array(kmer_dict[kmer])
        new_kmer_dict[kmer] = (np.mean(kmer_val_list), np.std(kmer_val_list))

    return new_kmer_dict

#############################################################


### Splitting Data into Train/Sample ###
def split_data(split, dict_dpts):
    random.seed(20)
    
    read_ids = list(dict_dpts.keys())
    indices = np.random.permutation(len(read_ids)).tolist()
    
    train_indices = random.sample(indices, split[0])
    indices = [i for i in indices if i not in train_indices]
    sample_indices = random.sample(indices, split[1])
    
    train_read_ids = [read_ids[i] for i in range(len(train_indices))]
    sample_read_ids = [read_ids[i] for i in range(len(sample_indices))]
    
    train_fast5_data =  [(read_id, dict_dpts[read_id][0]) for read_id in train_read_ids]
    sample_fast5_data = [(read_id, dict_dpts[read_id][0]) for read_id in sample_read_ids]
    
    train_fasta_data = [(read_id, dict_dpts[read_id][1]) for read_id in train_read_ids]
    sample_fasta_data = [(read_id, dict_dpts[read_id][1]) for read_id in sample_read_ids]
    
    return train_fasta_data, train_fast5_data, sample_fasta_data, sample_fast5_data 

##### Saving Data ####

def normalize_signal_data(data_list):
    data_list = [fast5_sig for read_id, fast5_sig in data_list]
    longest_read_length = max([len(read) for read in data_list])
    fast5_npy_array = np.zeros((len(data_list), 1, longest_read_length))
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            #print(i, j, fast5_npy_array[i,j], fast5_data[i][j])
            fast5_npy_array[i, 0, j] = data_list[i][j]
    return fast5_npy_array

def fast5_tuple_to_list(list_of_tuples):
    max_read_len = 0
    for read_id, fast5 in list_of_tuples:
        if len(fast5) > max_read_len:
            max_read_len = len(fast5)
    
    fast5_list = []
    for idx, (read_id, fast5_data) in enumerate(list_of_tuples):
        if len(fast5_data) < max_read_len:
            fast5_data = fast5_data + [0]*(max_read_len - len(fast5_data))
        fast5_list += [fast5_data]

    return fast5_list


def fast5_dict_to_list(data_dict):
    print('dict to list')
    read_id_data = []
    max_read_len = 0
    for read_id in data_dict:
        if len(data_dict[read_id][0]) > max_read_len:
            max_read_len = len(data_dict[read_id][0])
        read_id_data += [(read_id, data_dict[read_id][0])]
    
    print('adding values')
    
    fast5_list = []
    for idx, (read_id, fast5_data) in enumerate(read_id_data):
        if len(fast5_data) < max_read_len:
            fast5_data = fast5_data + [0]*(max_read_len - len(fast5_data))
        fast5_list += [fast5_data]

    return fast5_list


def trim_zeros(data):
    last = len(data)
    for idx, val in enumerate(data[:last][::-1]):
        if (val < 1 and val >= 0) or (val > -1 and val <= 0):
            last -= 1
        else:
           break
    return data[:last]

def trim_all_zeros(data_list):
    new_data_list = []
    for data in data_list:
        new_data_list += [trim_zeros(data)]
    return new_data_list


def save_fast5_data(data_list, path):
    
    header, metadata = read_slow5("/home/raehash.shah/SIMON/model/example.slow5")
    
    temp_data_list = data_list
    index = 0
    while len(temp_data_list) > 0:
        write_slow5(header, metadata, trim_all_zeros(temp_data_list[:4000]), path + f"slow5_{index}.slow5")
        convert_slow5_to_fast5(path + f"slow5_{index}.slow5", path + f"fast5_{index}.fast5")
        temp_data_list = temp_data_list[4000:]
        index += 1
    
    return 


def fasta_tuple_to_list(list_of_tuples):
    fasta_dict = dict()
    for read_id, fasta in list_of_tuples:
        fasta_dict[read_id] = fasta
    
    return fasta_dict


def save_fasta_data(data_dict, path):
    with open(path + "All_read.fasta", "w+") as fasta:
        for read_id in data_dict:
            fasta.write(str(read_id) + "\n")
            fasta.write(data_dict[read_id] + "\n")
        fasta.close()
    return



