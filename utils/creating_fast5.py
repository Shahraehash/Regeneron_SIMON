import os
import pyslow5
import uuid
import numpy as np


def convert_fast5_to_slow5(slow5_path):
    BASIC_FAST5 = "/home/raehash.shah/SIMON/model/example.fast5"
    os.system(f"slow5tools f2s {BASIC_FAST5} -o {slow5_path}")
    return

def read_slow5(filename):
    s5 = pyslow5.Open(filename, 'r')
    with open(filename, 'r') as slow5:
        header = [next(slow5) for _ in range(49)]
        
    list_of_metadata = {}
    list_of_metadata['read_group'] = []
    list_of_metadata['digitisation'] = []
    list_of_metadata['offset'] = []
    list_of_metadata['range'] = []
    list_of_metadata['sampling_rate'] = []
    list_of_metadata['start_time'] = []
    list_of_metadata['read_number'] = []
    list_of_metadata['start_mux'] = []
    list_of_metadata['median_before'] = []
    list_of_metadata['end_reason'] = []
    list_of_metadata['channel_number'] = []
    
    all_reads = s5.seq_reads(aux = 'all')
    for read in all_reads:
        list_of_metadata['read_group'] += [read['read_group']]
        list_of_metadata['digitisation'] += [read['digitisation']]
        list_of_metadata['offset'] += [read['offset']]
        list_of_metadata['range'] += [read['range']]
        list_of_metadata['sampling_rate'] += [read['sampling_rate']]
        list_of_metadata['start_time'] += [read['start_time']]
        list_of_metadata['read_number'] += [read['read_number']]
        list_of_metadata['start_mux'] += [read['start_mux']]
        if np.isnan(read['median_before']):
            list_of_metadata['median_before'] += [0.0]
        else:
            list_of_metadata['median_before'] += [read['median_before']]
        list_of_metadata['end_reason'] += [read['end_reason']]
        list_of_metadata['channel_number'] += [read['channel_number']]
        
    return header, list_of_metadata

def remove_nan(data):
    new_data = []
    for elem in data:
        if isinstance(elem, str):
            if 'nan' in elem:
                print(elem)
                new_data += [0]
        else:
            new_data += [elem]
    return new_data


def write_slow5(header, metadata, signal_data, new_slow5):
    with open(new_slow5, 'w+') as slow5:
        slow5.writelines(header)
        for idx, signal in enumerate(signal_data):
            read_id = str(uuid.uuid4())
            signal_string = ','.join(str(sig) for sig in signal)
            
            output_data = [read_id, metadata['read_group'][idx], metadata['digitisation'][idx], metadata['offset'][idx],
                           metadata['range'][idx], metadata['sampling_rate'][idx], len(signal), signal_string, 
                           metadata['start_time'][idx], metadata['read_number'][idx], metadata['start_mux'][idx], 
                           metadata['median_before'][idx], metadata['end_reason'][idx], metadata['channel_number'][idx]]
            
            output_data_new = [str(item) for item in output_data] 
            
            joined_output = '\t'.join(output_data_new) + '\n'
            
            slow5.write(joined_output)
    
    return

def convert_slow5_to_fast5(slow5, fast5):
    os.system(f"slow5tools s2f {slow5} -o {fast5}")
    return
