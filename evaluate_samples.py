from utils.evaluation_utils import *


def read_length_distribution(real_fast5, pseudo_fast5, num):
    #Choose random number of samples for read length and just store read lengths
    real_fast5_reads, max_real = get_fast5_data(real_fast5)
    pseudo_fast5_reads, max_pseudo = get_fast5_data(pseudo_fast5)
    real_idx = np.random.choice(len(real_fast5_reads), num)
    psuedo_idx = np.random.choice(len(pseudo_fast5_reads), num)
    real_fast5_reads = [real_fast5_reads[i] for i in real_idx]
    pseudo_fast5_reads = [pseudo_fast5_reads[i] for i in psuedo_idx]
    true_read_lengths = [len(read) for read in real_fast5_reads]
    psuedo_read_lengths = [len(read) for read in pseudo_fast5_reads]

    #Plot read lengths
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.hist([psuedo_read_lengths, true_read_lengths], label = ["Pseudo Reads", "True Reads"])
    ax.set_xlabel("Read Length")
    ax.set_ylabel("Occurences")
    ax.set_title("Distribution of Read Lengths")
    plt.legend()
    fig.savefig("results/ReadLengthDistribution.png") 
    return    

def classifier_fast5(real_fast5, pseudo_fast5, num):
    #Choose random number of datapoints
    real_fast5_reads, max_real = get_fast5_data(real_fast5)
    pseudo_fast5_reads, max_pseudo = get_fast5_data(pseudo_fast5)
    real_idx = np.random.choice(len(real_fast5_reads), num)
    psuedo_idx = np.random.choice(len(pseudo_fast5_reads), num)
    
    #Merge to form a numpy array with datalabels as well
    real_fast5_reads = [real_fast5_reads[i] for i in real_idx]
    pseudo_fast5_reads = [pseudo_fast5_reads[i] for i in psuedo_idx]
    real_fast5_labels = np.zeros((1,len(real_fast5_reads)))
    pseudo_fast5_labels = np.ones((1,len(pseudo_fast5_reads)))
    npy_data = merge_real_pseudo(real_fast5_reads, max_real, pseudo_fast5_reads, max_pseudo)
    data_labels = list(np.hstack((real_fast5_labels, pseudo_fast5_labels))[0])
    
    #Split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(npy_data, data_labels, train_size = 0.6, test_size = 0.4, random_state = 13)
    classifier = LogisticRegression(max_iter=10000)
    
    #Training fit and compute accuracy
    classifier.fit(x_train, y_train)
    predict_train = classifier.predict(x_train)
    train_accuracy = accuracy_score(y_train, predict_train)
    
    #Testing accuracy
    predict_test = classifier.predict(x_test)
    test_accuracy = accuracy_score(y_test, predict_test)
    
    return train_accuracy, test_accuracy

def pca_plot(real_fast5, pseudo_fast5, num, filename):
    #Choose random number of datapoints
    real_fast5_reads, max_real = get_fast5_data(real_fast5)
    pseudo_fast5_reads, max_pseudo = get_fast5_data(pseudo_fast5)
    real_idx = np.random.choice(len(real_fast5_reads), num)
    psuedo_idx = np.random.choice(len(pseudo_fast5_reads), num)
    
    #Merge to form a numpy array with datalabels as well
    real_fast5_reads = [real_fast5_reads[i] for i in real_idx]
    pseudo_fast5_reads = [pseudo_fast5_reads[i] for i in psuedo_idx]
    real_fast5_labels = np.zeros((1,len(real_fast5_reads)))
    pseudo_fast5_labels = np.ones((1,len(pseudo_fast5_reads)))
    npy_data = merge_real_pseudo(real_fast5_reads, max_real, pseudo_fast5_reads, max_pseudo)
    data_labels = list(np.hstack((real_fast5_labels, pseudo_fast5_labels))[0])

    #Perform Scaler Fit on the numpy array
    scaler = StandardScaler()
    npy_data = scaler.fit_transform(npy_data)

    #Plot PCA plot of datapoints
    pca = PCA()    
    components = pca.fit_transform(npy_data)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot = plt.scatter(components[:,0], components[:,1], c = data_labels)
    plt.xlabel("PCA Component 0")
    plt.ylabel("PCA Component 1")
    plt.title("PCA plot of Psuedo Reads and Real Reads")
    plt.legend(handles=plot.legend_elements()[0], labels=["Real", "Pseudo"])
    plt.savefig(filename)
    return

def dynamictimewarp_fast5(real_fast5, pseudo_fast5):
    #Choose random number of datapoints
    real_fast5_reads, _ = get_fast5_data(real_fast5)
    pseudo_fast5_reads, _ = get_fast5_data(pseudo_fast5)
    real_idx = np.random.choice(len(real_fast5_reads), 25)
    psuedo_idx = np.random.choice(len(pseudo_fast5_reads), 25)
    real_fast5_reads = [real_fast5_reads[i] for i in real_idx]
    pseudo_fast5_reads = [pseudo_fast5_reads[i] for i in psuedo_idx]
    
    #Compute dynamic time warping distance and normalize the values based on average distance so only the change
    dtw_dist = []
    for idx1 in range(len(real_fast5_reads)):
        read1_dtw_dist = []
        for idx2 in range(len(pseudo_fast5_reads)):
            read1 = real_fast5_reads[idx1]
            read2 = pseudo_fast5_reads[idx2]
            read1_dtw_dist += [fastdtw_process(read1, read2)[0]]
        average_pseudo = sum(read1_dtw_dist)/len(read1_dtw_dist)
        updated_read1_dtw_dist = []
        for read in read1_dtw_dist:
            updated_read1_dtw_dist += [read/average_pseudo]
        dtw_dist += [updated_read1_dtw_dist]
    
    #Save DTW values in .txt file
    np.savetxt("results/DTW_values.txt", np.array(dtw_dist))
    
    #Plot DTW Distance 
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.imshow(dtw_dist)
    plt.colorbar()
    plt.xlabel("Real Reads")
    plt.ylabel("Pseudo Reads")
    plt.title("Heatmap of DTW Distance")
    fig.savefig("results/DTW_heatmap.png")
    return 

def levenshtein_fastq(true_fasta, pseudo_fastq, num):
    true_fasta_list = translate_fasta(true_fasta)
    pseudo_fastq_list = translate_fastq(pseudo_fastq)
    
    true_idx = np.random.choice(len(true_fasta_list), num)
    pseudo_idx = np.random.choice(len(pseudo_fastq_list), num)
    
    true_fasta_list = [true_fasta_list[i] for i in true_idx]
    pseudo_fastq_list = [pseudo_fastq_list[i] for i in pseudo_idx]

    lev_dist_vals = []
    for true in true_fasta_list:
        true_lev_dist = []
        for pseudo in pseudo_fastq_list:
            # normalized so it's the number of mismatches, indels for the longest one to match the shorter one
            true_lev_dist += [levenshtein_distance(true, pseudo)/max(len(true), len(pseudo))] 
    
        lev_dist_vals += [true_lev_dist]
    
    np.savetxt("results/Levenshtein_values.txt", np.array(lev_dist_vals))

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.imshow(lev_dist_vals)
    plt.colorbar()
    plt.xlabel("Real Reads")
    plt.ylabel("Pseudo Reads")
    plt.title("Heatmap of Levenshtein Distance")
    plt.xlim(0, num)
    plt.ylim(0, num)
    fig.savefig("results/Levenshtein_heatmap.png")
    
    return


def coverage_map(generated_fastq, reference_genome):
    fastqs = ""
    for fastq in os.listdir(generated_fastq):
        if fastq.endswith('.fastq'):
            fastqs += str(os.path.join(generated_fastq, fastq)) + " "
    
    
    os.system(f"minimap2 -k 9 -B 1.5 -ax map-ont {reference_genome} {fastqs} > output.sam")
    os.system("samtools sort output.sam > output.bam")
    os.system("samtools depth -a output.bam > output.txt")
    
    data_file = "output.txt" #"/data/Raehash_SSSD/data/CHO_data/generated_data/output.txt"

    with open(data_file, "r") as f:
        lines = f.readlines()
    f.close()
    non_zeros = []
    for idx, line in enumerate(lines):
        if int(line.split('\t')[2]) != 0:
            non_zeros.append(idx)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.hist(non_zeros)
    ax.set_xlabel("Position on Reference Genome")
    ax.set_ylabel("Coverage of Genomic Position")
    ax.set_title("Coverage of Generated Reads on Reference Genome")
    fig.savefig("results/CoverageMap.png") 
    
    return



if __name__ == "__main__":
    random.seed(100)
    sample_fast5_folder = "/data/Raehash_SSSD/data/CHO_data/train/"
    generated_fast5_folder = "/data/Raehash_SSSD/data/CHO_data/generated_data/"
    
    #Plot Read Length Distributions
    print("Read Length Distribution of Sample and Generated Data")
    read_length_distribution(sample_fast5_folder, generated_fast5_folder, 1000)
    
    
    #Binary Classifier + PCA
    print("Classifier Accuracy of Sample and Generated Data")
    train_accuracy, test_accuracy = classifier_fast5(sample_fast5_folder, generated_fast5_folder, 30)
    print("Training Accuracy: ", train_accuracy, "Testing Accuracy: ", test_accuracy)
    pca_plot(sample_fast5_folder, generated_fast5_folder, 100, "results/PCA_plot.png")
    
    #Dynamic Time Warping Distance
    print("Computing Dynamic Time Warping Distance")
    dynamictimewarp_fast5(sample_fast5_folder, generated_fast5_folder)
    
    
    #Basecall Reads
    os.system(f"guppy_basecaller -i {generated_fast5_folder} -s {generated_fast5_folder} -c dna_r9.4.1_450bps_hac.cfg")
    
    sample_fasta = "/data/Raehash_SSSD/data/CHO_data/sample/All_read.fasta"
    generated_fastq = "/data/Raehash_SSSD/data/CHO_data/generated_data/fail/"
    Reference_fasta = "/data/Raehash_SSSD/data/REQUESTED_DATA/CHO.fasta"
    
    #Levenshtein Distance
    print("Computing Levenshtein Distance")
    levenshtein_fastq(sample_fasta, generated_fastq, 25)
    
    #Coverage Mapping
    print("Getting coverage map")
    coverage_map(generated_fastq)