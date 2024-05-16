# A machine learning approach for computing solar flare locations in X-rays on-board Solar Orbiter/STIX

This repository contains the dataset and the code developed for addressing the problem of estimating solar flare locations from data recorded by the Coarse Flare Locator (CFL) 
on board the Spectrometer/Telescope for Imaging X-rays (STIX) with a neural network. 
The work is described in a paper that will appear soon on arxiv.

# Dataset
The [dataset](dataset) folder contains the training, validation and test set files. Each file contains the following information:
* nn_data: array of dimension N_events x 9 containing the CFL data that are given as input to the MLP
* stx_flare_loc: array of dimension N_events x 2 containing the X and Y coordinates of the flares' location derived from STIX imaging information
* cfl_loc: array of dimension N_events x 2 containing the X and Y coordinates of the flares' location estimated by the CFL algorithm
* start_time: array of dimension N_events containing the start observation time of each example
* end_time: array of dimension N_events containing the end observation time of each example
* tot_counts: array of dimension N_events containing the total number of counts registered by the STIX imaging detectors for each example
* tot_counts_cfl: array of dimension N_events containing the total number of counts registered by the STIX CFL detector for each example
* sidelobes_ratio: array of dimension N_events an estimate of the reliability of the flare location derived from STIX imaging (ratio between the peak flux and the sidelobes' flux)
* file_uid: array of dimension N_event containing the UID of the STIX FITS file from which the data are extracted 

# Model definition and training
The [train_nn.ipynb](train_nn.ipynb) notebook shows how the neural network is defined and trained. Auxiliary functions used for defining the model and for normaliziong the dataset contained in [aux_code.py](aux_code.py).

# Quantized model predictions on the test set
The [test_quantized_nn.ipynb](test_quantized_nn.ipynb) notebook shows how to quantized the neural network inputs, weights and biased in such a way that all the operations in the forward pass are performed in 16-bit fixed-point integer arithmetic. The operations performed by the quantized neural network are implemented in [nn_model](https://github.com/paolomassa/STX_CFL_NN/blob/a5fc18334ae50416a1f550097626fd2027243cab/aux_code.py#L107C5-L107C13). A comparison with the flare locations estimates provided by the CFL algorithm is also shown in the notebook.
