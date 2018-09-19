# CNCE code used for ICML 2018 results

This code reproduces the results of the ICML 2018 paper "Conditional Noise-Contrastive Estimation". 

Posible Python implementation might appear in the future. 

For the synthetic dataset where the ground truth is available the results are not quantitatively exactly the same as those of the ICML paper
as the seed used to generate the ground truth parameters was lost. 
Furthermore, the currently configuration only reproduces the results partially to speed up the computations.
The full result can be obtained by modifiying the config files.
In particular the optimisation option 'opt.alg' should be set to 'all'.
With is option both Matlab's 'funminunc' and the 'minimize.m' optimisation is used and the estimate with the lowest lost is reported.
This was important for NCE which seemed to be senitive to the optimiser (sometimes matlabs 'funmiunc' worked failed and sometimes the 'minimize.m' failed. )



