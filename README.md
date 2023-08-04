# SIMON (SIgnal siMulated by diffusiON) - A Novel Diffusion Machine Learning Model that Accurately Simulates ONT Signal Data

### By Raehash Shah

---

## Purpose: 

SIMON is a conditional diffusion based machine learning model which generates large amounts of ONT signal data from a nucleotide sequence. This tool will help provide adequate amounts of data for the QC Virology Development team to generate tools to help supplement current detection protocols for viruses. Currently SIMON has been shown to perform well in generating sequencing data for CHO (mammalian) sequencing data. 

---

## Background:

In the past decade, nanopore sequencing has become the preferred method for generating accurate long reads in real time. As the DNA molecule passes through the biological nanopore of an Oxford Nanopore Technologies (ONT) sequencer, the sequencer reads the electrical current signal of the nanopore and translates the signal data into the sequence of the DNA molecule. As more applications rely on having large amounts of sequencing data, a tool is proposed that can accurately simulate signal data. SIMON (SIgnal siMulated by diffusiON) is a novel method that uses a conditional diffusion model and structured state space model to generate signal data. This model is inspired and built on top of the "Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models" paper found here: https://openreview.net/pdf?id=hHiIbk7ApW, which accurately simulated ECG data. Once SIMON generated signal data, signal and sequence evaluation metrics were used to evaluate SIMON. From these evaluation metrics, SIMON's signal data was almost indistinguishable from true signal data however, after basecalling the signal data, there is still some room for improvements on the accuracy of the corresponding reads generated from SIMON. In addition, SIMON has many rooms for further applications which include performing a comparison analysis and generating sequencing data for a spiked cell culture. 

---

## Requirements:

Data - The data for this model that was used was CHO and MVM fast5 and fasta data that can be found on the promethION /data/Raehash_SSSD/data/REQUESTED_DATA.
Computational - All code was written and run on the ONT PromethION. This device had an OS system of Ubuntu 20.04, 4 x A100 NVIDIA GPUs, CUDA version of 11.7 and with a Conda verison of 23.5.2. 

---

## Installation:

All packages can be found in requirements.yml and requirements.txt

---

## Usage:
 
After all package requirements have been setup, there is access to the necessary computational and data requirements, SIMON can be prepared. 

### Pre-processing:

Input: (1) Metadata file of read_id to organism, (2) Folder of .fast5 pass files and (3) Folder of .fastq pass files
Output: (1) Folder of training data (.fast5 files and .fasta file), (2) Folder of sampling data (.fast5 files and .fasta file) and (3) kmer-to-signal database .txt file, (4) Graphs of read length distributions (x3)
In pre-processing, the data is setup so that it is usable for the downstream training and sampling and a kmer database is created according to the dataset that SIMON is currently working with. Note that split of training and sampling can be adjusted within preprocessing.py. In addition, to further understand the dataset that is being worked with, the distribution of read lengths are plotted and if there is any filtering of longer reads, the respective distribution is also plot with and without the best fit classical distribution that best explains the read length distribution.


**python3 preprocessing.py --metadata {path to metadata file} --fast5 {path to .fast5 pass folder} --fastq {path to .fastq pass folder}**

### Training:

Input: (1) Folder of .fast5 training data 
Output: (1) Training mean-squared error loss graph (x2), (2) ideal diffusion hyperparameters
After pre-processing is performed, or a set of .fast5 signal data has been decided to use train the model, training of SIMON can be performed. The input folder of .fast5 data is translated to a readable numpy array which can be passed into the model. Then hyperparameter tuning is performed to identify the ideal diffusion hyperparameters for SIMON (note that the range of exploration can be adjusted to finetune or more broadly explore the state space). Please be careful in terms of storage capabilities when adjusting this since each unique pair of hyperparameters correspond to a new iteration of models which can become really large and may take up the space available on your device. Once ideal hyperparamets have been decided, the model is then loaded and those are the hyperparameters outputted. In addition, the mean-squarred error over the number of iterations is plotted to show the decrease in loss over time (the last 50 iterations are also shown in a separate plot to highlight the specific differences over a shorted time period). 

**python3 train_model.py --fast5 {path to .fast5 folder of training from pre-processing}**

### Sampling:

Input: (1) File of .fasta sampling data, (2) Declaration of number of samples that are desired (default: 3 * 4000 samples) 
Output: (1) Folder of .fast5 files
Once an ideal model has been decided we can generate samples as desired. Once input .fasta file is passed into the model and a desired number of samples is provided, the model will begin genreating samples according to its training knowledge. PLEASE ADJUST THE HYPERPARAMETERS SO THEY ARE THE IDEAL ONES FROM TRAINING. Once sampling is complete, post-processing is complete to trim zeros and add gaussian noise so they are more representative of true reads. These reads are then outputted into .fast5 files which can then be basecalled. 

**python3 generate_samples.py --fasta {path to .fasta file of sampling from pre-processing} --num_samples {number of .fast5 files desired (each number will be multiplied by 4000 to generate the respective number of files)}**

### Evaluation:

Input: (1) Sampling .fast5 folder, (2) Generated .fast5 folder, (3) Sampling .fasta file
Output: (1) Read Length Distribution Plot, (2) Binary Classifier Accuracy, (3) PCA plot, (4) Dynamic Time Warping Distance Heatmap plot, (5) Levenshtein Distance Heatmap plot, (6) Coverage Map Plot
To evaluate the signal data generated, a multitude of approaches will be taken to better understand how this simulated data compares to real sequencing data. One preliminary approach is to plot the distribution of read lengths of the real sequencing data and the simulated data. As shown, the graph shows SIMON does well in matching the distribution of ONT sequencing reads but significantly overestimates the number of shorter reads (~ 2000 bp). The next main approach is to evaluate the similarity of the raw signal values. To do this a binary classifier was trained on a subset of the real sequencing data and the generated sequencing data and then tested on another subset of the data. The binary classifier had a 46% accuracy which is good cause a trained model did worse than randomly assignment of labels suggesting the similarity between the signals. To confirm this, each signal was plotted on a PCA to show the higher dimensional data on a lower dimensionality plot. In addition, a dynamic time warping distance was generated between a subsample of both types of reads. This was then normalized to tell us how adjustments to the waveform were necessary to minimize the distance value. The results show this was low for most of the pairwise comparisons between reads. 

However signal value is useless without the signal data basecalled into reads which QC Virology can work with. Therefore using Guppy, all of the .fast5 data was basecalled.

**guppy_basecaller -i {input .fast5 folder} -s {destination for .fastq reads} -c {configuration of test run (ex: dna_r9.4.1_450bps_hac.cfg)}**

After basecalling, the reads were evaluated by performing a levenshtein distance to measure how many mismatches and indels were present between each pair of reads and normalized for the length of the longer read. The results suggest that although for every 3 basepairs, 2 operations (indels/mismatches) need to occur, this is still better than complete alterations which means that the reads can be representative of the desired species. In addition, the reads were mapped to a reference genome and the coverage map were plotted to see the number of reads present in each genomic position. 

**python3 evaluate_samples.py**

---

## Experimental Overview:

Find all powerpoint presentations and posters in the repository which were presented to QC Virology and the rest of IOPS Rensellear.

---

## Next Steps:

Some major avenues of future directions are to (1) improve SIMON's accuracy by implementing a better read distribution, increasing the size of SIMON, adding more training data to SIMON, better adjusting for the deviations in the signal data; (2) perform comparison analysis for read generation of CHO mammalian data and MVM viral data or long read generation vs short read generation; (3) a simulation model of both CHO mixed with MVM at a given concentration read generation. 


