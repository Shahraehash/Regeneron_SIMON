import argparse, os
from utils.sampling_utils import *

def generate_sample_pipeline(fasta_file, num_samples, diffusion_config, wavenet_config, train_config, trainset_config, gen_config):
    
    # Create mapping of sequence to pseudo signal and make them uniform to generate .npy
    print("Converting .fasta folder to .npy")
    npy_fasta, mean, stdev, max_sample_len = fasta_to_npy(fasta_file, wavenet_config["in_channels"])
    with open(fasta_file.split("All_read.fasta")[0] + "sampling_data.npy", 'wb+') as f:
        np.save(f, npy_fasta)
    

    train_config["mean"] = mean
    train_config["stdev"] = stdev 
    
    
    print("Generating Samples")
    # Load the model and pass the .npy through the model to generate one sample
    mse_list, output_path = generate_sample(num_samples, max_sample_len, diffusion_config, wavenet_config, train_config, trainset_config, gen_config)
    #output_path = "/data/Raehash_SSSD/results/CHO_data/T200_beta00.0001_betaT0.02_rm" + "/all_generated_data_0.npy"


    # Add post-processing to add noise, dynamic time warping and to trim 0s
    print("Perform Post-Processing and save signal data to ", fasta_file.split('sample')[0] + "generated_data/")
    npy_output = np.load(output_path)
    signal_data = [signal[0].tolist() for signal in npy_output]
    
    if not os.path.isdir(fasta_file.split('sample')[0] + "generated_data/"):
        os.mkdir(fasta_file.split('sample')[0] + "generated_data/")
    
    save_fast5(signal_data, fasta_file.split('sample')[0] + "generated_data/")
    
    return



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    
    parser = argparse.ArgumentParser(description = "Inputs for Generating samples using the model")
    parser.add_argument('--fasta', type = str, help = "Pass path for fasta file")
    parser.add_argument('--num_samples', type = str, default = 3, help = "Provide number of fast5 you want to generate")
    
    args = parser.parse_args()
    fasta_file = args.fasta
    
    if not os.path.isfile(fasta_file):
        raise Exception('Path for .fasta file provided is invalid')
    
    #Ideal hyper parameters based on Training
    diffusion_config = {"T": 200, "beta_0": 0.0001, "beta_T": 0.02} 
    learning_rate = 0.0002
    missing_k = 90
    
    #Wavenet config
    wavenet_config = {"in_channels": 13164, "out_channels": 13164, "num_res_layers": 36, 
                      "res_channels": 256, "skip_channels": 256, "diffusion_step_embed_dim_in": 128, 
                      "diffusion_step_embed_dim_mid": 512, "diffusion_step_embed_dim_out": 512, "s4_lmax": 100, 
                      "s4_d_state":64, "s4_dropout":0.0, "s4_bidirectional":1, "s4_layernorm":1}
    
    train_config = {"ckpt_iter": "max", "iters_per_ckpt": 100, "learning_rate": learning_rate, "only_generate_missing": 1, "missing_k": missing_k, "mean": 0, "stdev": 1}
    testing_data = "/data/Raehash_SSSD/data/CHO_data/sample/sampling_data.npy"
    gen_config = {"output_directory": "/data/Raehash_SSSD/results/CHO_data/", "ckpt_path": "/data/Raehash_SSSD/results/CHO_data/"}
    
    
    generate_sample_pipeline(fasta_file, int(args.num_samples)*4000, diffusion_config, wavenet_config, train_config, testing_data, gen_config)