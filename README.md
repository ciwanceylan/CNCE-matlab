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

### Synthetic data
- Use matlab/synthetic_data as the "Current Folder" in Matlab
- Run `setPath_synthetic_data` to set the code path and add global variables specifying
data and results folder.
- In bin/experiments the code for reproducing the results is found:
    - The `run_all_*` scripts will run the desired experiment. 
    To also plot the results some lines need to be uncommented.
    - The default setup is to only run the experiments partially so it doesn't take so long.
    This can be modified in the `config_*` files.
- For further details see the commented code.

**NOTE:** Unfortunately the seed which was used to generate the ground truth 
parameters used for the results in the paper has been lost. So reproduced results 
may vary slightly.
Furthermore, when NCE is used to estimate the ICA model the optimisation is sensitive 
to the choice of optimiser. For the results in the paper we therefore optimised 
using two different optimisers and chose the parameter value with the lower loss.
This is not done by default in this code, so the results for NCE with ICA model 
will look worse than in the paper. This can be changed by setting `setup.optNCE.alg`
in the config files to `'all'`.


### Natural images
- Use matlab/natural_images as the "Current Folder" in Matlab
- Run `setPath_natural_images` to set the code path and add global variables specifying
data and results folder.


- In the subfolder
For the synthetic dataset where the ground truth is available the results are not quantitatively exactly the same as those of the ICML paper
as the seed used to generate the ground truth parameters was lost. 
Furthermore, the currently configuration only reproduces the results partially to speed up the computations.
The full result can be obtained by modifiying the config files.
In particular the optimisation option 'opt.alg' should be set to 'all'.
With is option both Matlab's 'funminunc' and the 'minimize.m' optimisation is used and the estimate with the lowest lost is reported.
This was important for NCE which seemed to be senitive to the optimiser (sometimes matlabs 'funmiunc' worked failed and sometimes the 'minimize.m' failed. )



