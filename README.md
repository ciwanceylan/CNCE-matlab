# Code for Conditional Noise-Contrastive Estimation of Unnormalised Models, presented at ICML 2018
[paper](https://arxiv.org/abs/1806.03664 "Conditional Noise-Contrastive Estimation of Unnormalised Models, ICML 2018")

This repo contains the code for reproducing the results presented in the ICML 2018 
paper _Conditional Noise-Contrastive Estimation of Unnormalised Models_.
The code is implemented in Matlab and tested for version 2018a but should work 
for older versions as well. The code for the synthetic data experiments and the 
natural image experiments lie in seperate folders.

Any publication that discloses findings arising from using this source code must 
cite “Conditional Noise-Contrastive Estimation of Unnormalised Models", PMLR 80:726-734 (2018)

## Usage

For the synthetic dataset where the ground truth is available the results are not quantitatively exactly the same as those of the ICML paper
as the seed used to generate the ground truth parameters was lost. 
Furthermore, the currently configuration only reproduces the results partially to speed up the computations.
The full result can be obtained by modifiying the config files.
In particular the optimisation option 'opt.alg' should be set to 'all'.
With is option both Matlab's 'funminunc' and the 'minimize.m' optimisation is used and the estimate with the lowest lost is reported.
This was important for NCE which seemed to be senitive to the optimiser (sometimes matlabs 'funmiunc' worked failed and sometimes the 'minimize.m' failed. )



