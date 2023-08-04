import os
import numpy as np
from Levenshtein import distance as lev_distance
from ont_fast5_api.fast5_interface import get_fast5_file
from fastdtw import fastdtw
from scipy.spatial import distance
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import random

### Evaluating DTW of Fast5 Files ###
def get_fast5_data(path):
    fast5_data = []
    longest_read_length = -1 * float("inf")    
    for file in os.listdir(path):
        if file.endswith('.fast5'):
            with get_fast5_file(os.path.join(path, file), mode = "r") as f5:
                for read in f5.get_reads():
                    fast5_data += [read.get_raw_data()]
                    if len(read.get_raw_data()) > longest_read_length:
                        longest_read_length = len(read.get_raw_data())
    
    return fast5_data, longest_read_length

def fastdtw_process(read1, read2):
    read1_x = [i for i in range(len(read1))]
    read2_x = [j for j in range(len(read2))]
    
    read1_new = np.hstack((np.array(read1_x).reshape(len(read1_x), 1), np.array(read1).reshape(len(read1), 1)))
    read2_new = np.hstack((np.array(read2_x).reshape(len(read2_x), 1), np.array(read2).reshape(len(read2), 1)))
    
    dtw_distance, warp_path = fastdtw(read1_new, read2_new, dist = distance.minkowski)
    
    return dtw_distance, warp_path 

### Classifier for Fast5 Files ###
def merge_real_pseudo(fast5_read1, max_len_read1, fast5_read2, max_len_read2):
    npy_data = np.zeros((len(fast5_read1) + len(fast5_read2), max(max_len_read1, max_len_read2)))
    
    for i in range(len(fast5_read1)):
        for j in range(len(fast5_read1[i])):
            npy_data[i,j] = fast5_read1[i][j]
    
    for i in range(len(fast5_read2)):
        for j in range(len(fast5_read2[i])):
            npy_data[i+len(fast5_read1), j] = fast5_read2[i][j]

    return npy_data 



### Evaluating Levenshtein Fastq Files ###
def translate_fasta(path):
    fasta_list = []
    if path.endswith('.fasta'):
        with open(path, "r") as f:
            fasta_data = f.readlines()
    
    for idx, line in enumerate(fasta_data):
        if idx % 2 == 1:
            fasta_list += [line[:-1]]
    return fasta_list

def translate_fastq(folder):
    lines = []
    for filename in os.listdir(folder):
        if filename.endswith('.fastq'):
            with open(os.path.join(folder, filename), "r") as fq: 
                lines += fq.readlines()
            fq.close()
    
    fastq_list = []
    for idx in range(int(len(lines)/4)):
        fastq_list += [lines[4*idx+1][:-1]]
    
    return fastq_list
    

def levenshtein_distance(s1, s2):
    return lev_distance(s1, s2)


### Signal Alignment ###
