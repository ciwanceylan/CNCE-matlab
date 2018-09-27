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
- In bin/training_scripts three scripts which trains the model is found:
    - `run_full_CNCE_training` trains the full model (all four layers) with CNCE
    using 32x32 patches and 600 PCA dimensions.
    This took me between 1 to 2 days on my CPU (not that no GPU training code is 
    included). 
    - `run_small_training` also trains the full model, but using 25x25 patches and
    160 PCA dimensions. It is therefore faster and can be trained in a few hours.
    - `run_partial_CNCE_NCE_training` trains the first and second layer using 
    32x32 patches and 600 PCA dimensions, but with fewer neurons. This was required
    to train this model using NCE as the optimisers had issues when more neurons 
    where used. This code is mainly used to obtain checkpoints from which the feature
    comparision seen in Figure 2 to 13 in the Supplementary Material can be generated
    (using `make_CNCE_NCE_gif` in the "visualisation" folder).
- The data will be downloaded automatically  at the start of training.
- The code will also generate the PCA transformation matricies and store them locally.
- After training the results can be visualised using `plot_receptive_fields` in the 
folder "visualisation". Note that the space-orientation receptive field visualisations 
for layer 3 and 4 can be slow.

**NOTE:** As the original 11 Van-Hateren image used for training cannot be downloaded
seperately, the automatic download instead downloads 11 random images from the same 
dataset. If you want to download the whole dataset yourself you can find it here:
[Van Hateren images](http://pirsquared.org/research/vhatdb/full/).


