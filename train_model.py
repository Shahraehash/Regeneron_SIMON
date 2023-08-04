import argparse, os
import matplotlib.pyplot as plt
from utils.training_utils import *

def train_model(fast5_folder):
    #Convert all fast5_files in fast5_folder to a single .npy 
    print("Converting .fast5 folder to .npy")
    fast5_npy_array, mean, stdev, longest_read_length = fast5_to_npy(fast5_folder)
    
    with open(fast5_folder + "training_data.npy", 'wb+') as f:
        np.save(f, fast5_npy_array)
    

    # SSSD-S4 set configurations
    train_config = {"output_directory": "/data/Raehash_SSSD/results/CHO_data/", "ckpt_iter": "max", "iters_per_ckpt": 100,
                    "iters_per_logging": 100, "n_iters": 10000, "learning_rate": 2e-4, "only_generate_missing": 1,
                    "missing_k": 90, "mean": mean, "stdev": stdev}
    trainset_config = {"train_data_path": fast5_folder + "training_data.npy",
                       "segment_length":100, "sampling_rate": 100}
    diffusion_config = {"T": 200, "beta_0": 0.0001, "beta_T": 0.02}
    wavenet_config = {"in_channels": longest_read_length, "out_channels": longest_read_length, "num_res_layers": 36, #make sure 
                      "res_channels": 256, "skip_channels": 256, "diffusion_step_embed_dim_in": 128,
                      "diffusion_step_embed_dim_mid": 512, "diffusion_step_embed_dim_out": 512, "s4_lmax": 100,
                      "s4_d_state":64, "s4_dropout":0.0, "s4_bidirectional":1, "s4_layernorm":1}
    
    
    
    #Pass npy array into model + save loss values
    print("Hyperparameter tuning")
    T_range = list(range(150, 250, 10)) #10 iterations
    beta0_range = np.arange(1e-04, 1e-03, 1e-04).tolist() #10 iterations
    betaT_range = np.arange(1e-02, 1e-01, 1e-02).tolist() #10 iterations
    ideal_T, ideal_beta0, ideal_betaT = hyperparameter_tuning(T_range, beta0_range, betaT_range, train_config, trainset_config, diffusion_config, wavenet_config)
    
    
    #Training based on hyperparameter tuning
    print("Training on ideal hyperparameters")
    vals = training(train_config, trainset_config, diffusion_config, wavenet_config) 
    x_vals = []
    y_vals = []
    for (iteration, loss_value) in vals:
        x_vals += [iteration]
        y_vals += [loss_value]
        
    #Plot and Save loss values
    print("Plotting MSE loss values")
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(x_vals, y_vals)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"results/Training_loss_{ideal_T}_{ideal_beta0}_{ideal_betaT}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(x_vals[-51:], y_vals[-51:])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"results/Training_loss_{200}_{0.0001}_{0.02}_better.png")
    
    return ideal_T, ideal_beta0, ideal_betaT

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1,3"
    parser = argparse.ArgumentParser(description = "Inputs for Training the model")
    parser.add_argument('--fast5', type = str, default = "/data/Raehash_SSSD/data/CHO_data/train", help = "Pass path for a folder of input .fast5 files")
    
    args = parser.parse_args()
    fast5_folder = args.fast5
    
    if not os.path.isdir(fast5_folder):
        raise Exception('Path for .fast5 folder provided is invalid')
    
    ideal_T, ideal_beta0, ideal_betaT = train_model(fast5_folder)
    
    print("Training complete with T:", ideal_T, ", Beta0:", ideal_beta0, ", BetaT:", ideal_betaT)