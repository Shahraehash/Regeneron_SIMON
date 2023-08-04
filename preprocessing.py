from ont_fast5_api.fast5_interface import get_fast5_file
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from fitter import Fitter, get_common_distributions, get_distributions

from utils.preprocessing_utils import *

#get only CHO data or only MVM data
def readid_species(metadata_file):
    CHO_data = dict()
    MVM_data = dict()
    with open(metadata_file, "r") as f:
        lines = f.readlines()
    f.close()
    
    for line in lines:
        read_id, organism = line[:-1].split("\t")
        if organism == "CHO":
            CHO_data[read_id] = []
        if organism == "MVM":
            MVM_data[read_id] = []
    
    return CHO_data, MVM_data

#Map readid to signal data
def add_signal_data(dictionary_file, fast5_folder):
    new_dict = dict()
    read_lengths = []
    for fast5_file in os.listdir(fast5_folder):
        if fast5_file.endswith('.fast5'):
            with get_fast5_file(os.path.join(fast5_folder, fast5_file), mode = "r") as f5:
                for read in f5.get_reads():
                    read_id = read.get_read_id()
                    if read_id in dictionary_file:
                        new_dict[read_id] = read.get_raw_data().tolist()
                        read_lengths += [len(read.get_raw_data().tolist())]
    
    return new_dict, read_lengths

#Map readid to signal data and sequencing data
def add_sequence_data(dictionary_file, percentile, fastq_folder):
    lines = []
    for fastq_file in os.listdir(fastq_folder):
        if fastq_file.endswith(".fastq"):
            with open(os.path.join(fastq_folder, fastq_file), "r") as fq:
                lines += fq.readlines()
            fq.close()

    new_dict = dict()
    for idx in range(int(len(lines)/4)):
        read_id = lines[4*idx].split("runid")[0][1:-1]
        sequence = lines[4*idx+1][:-1]
        if read_id in dictionary_file and len(dictionary_file[read_id]) <= percentile:
            new_dict[read_id] = [dictionary_file[read_id], sequence]
        
    return new_dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Pre-processing for the model")
    parser.add_argument('--metadata', type = str, default = "/data/Raehash_SSSD/data/REQUESTED_DATA/Tailored_Data.txt", help = "Pass path for a .txt file describing read_id to CHO or MVM")
    parser.add_argument('--fast5', type = str, default = "/data/Raehash_SSSD/data/REQUESTED_DATA/fast5_pass/", help = "Pass path for folder with all .fast5 files")
    parser.add_argument('--fastq', type = str, default = "/data/Raehash_SSSD/data/REQUESTED_DATA/fastq_pass/", help = "Pass path for folder with all .fastq files")
    
    args = parser.parse_args()
    metadata_file = args.metadata
    fast5_pass_folder = args.fast5
    fastq_pass_folder = args.fastq
    
    print('Getting ReadID of Species')
    CHO_data_dictionary, MVM_data_dictionary = readid_species(metadata_file)
    print('Adding Signal Data')
    CHO_data_dictionary, read_lengths = add_signal_data(CHO_data_dictionary, fast5_pass_folder)
    
    print("Plotting Distribution of Read Lengths") 
    percentile = np.percentile(read_lengths, 95)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.hist(read_lengths, bins = 1000)
    plt.title("Distribution of Raw Reads")
    plt.vlines(percentile, ymin = 0, ymax = max(read_lengths))
    fig.savefig("results/Training_Read_Distribution_untrimmed.png")
    
    trimmed_reads = [x for x in read_lengths if x < percentile]
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.hist(trimmed_reads, bins = 1000)
    plt.title("Distribution of Trimmed Reads")
    fig.savefig("results/Training_Read_Distribution_trimmed.png")
    
    f = Fitter(trimmed_reads, distributions = ['gamma', 'lognorm', "beta", "burr", "norm", "alpha"])
    f.fit()
    distribution = f.get_best(method = 'sumsquare_error')
    
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.hist(trimmed_reads, bins = 1000)
    x_vals = np.linspace(min(trimmed_reads), max(trimmed_reads), 1000)
    pdf_values = st.burr.pdf(x_vals, distribution['burr']['c'], distribution['burr']['d'])
    plt.plot(x_vals, [(x * 1e10) for x in pdf_values], color = 'red')
    fig.savefig("results/Training_Read_Distribution_trimmed_plus_burr.png")
    
    print('Adding Sequencing Data')
    CHO_data_dictionary = add_sequence_data(CHO_data_dictionary, percentile, fastq_pass_folder)
    
    
    print("Creating Database of Signal to Kmer")
    create_database(CHO_data_dictionary)
    
    
    print("Splitting Data into Train/Sample")
    split = [int(0.8*len(CHO_data_dictionary)), int(0.2*len(CHO_data_dictionary))]
    train_fasta, train_fast5, sample_fasta, sample_fast5 = split_data(split, CHO_data_dictionary)
    
    ### For saving split data
    print("Saving Data")
    data_path = "/data/Raehash_SSSD/data/CHO_data/"
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    training_data_path = "/data/Raehash_SSSD/data/CHO_data/train/"
    if not os.path.isdir(training_data_path):
        os.mkdir(training_data_path)
    sampling_data_path = "/data/Raehash_SSSD/data/CHO_data/sample/"
    if not os.path.isdir(sampling_data_path):
        os.mkdir(sampling_data_path)
    
    if not os.path.isfile("/home/raehash.shah/SIMON/model/example.slow5"):
        convert_fast5_to_slow5("/home/raehash.shah/SIMON/model/example.slow5")
    
    
    print("Saving Training file to ", training_data_path)
    save_fasta_data(fasta_tuple_to_list(train_fasta), training_data_path)
    save_fast5_data(fast5_tuple_to_list(train_fast5), training_data_path)
    
    print("Saving Sampling file to ", sampling_data_path)
    save_fasta_data(fasta_tuple_to_list(sample_fasta), sampling_data_path)
    save_fast5_data(fast5_tuple_to_list(sample_fast5), sampling_data_path)
    
