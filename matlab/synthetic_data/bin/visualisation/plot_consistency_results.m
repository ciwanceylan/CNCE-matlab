function plot_consistency_results(setup, res)
%PLOT_CONSISTENCY_RESULTS Plots the consistency results given the result
%struct 'res' generated from run_cnce_experiment

global CNCE_RESULTS_FOLDER

% Figure saving setup
figure_save_folder = fullfile(CNCE_RESULTS_FOLDER, 'figures',...
    setup.modelName, 'consistency');
savefile_CNCE_consistency = ['CNCE_consistency_' num2str(setup.D) 'D'];
savefile_NCE_consistency = ['NCE_consistency_' num2str(setup.D) 'D'];
savefile_consistency_comparision = ['CNCE_NCE_comparision' num2str(setup.D) 'D'];

ylimits = getYlim(setup.modelName);
kappa_ind = [1, 2, 4];

% CNCE consistency
y = log10(cat(3, res.err_cnce(:, :, kappa_ind, 1), res.err_mle));
[ h, harr, hp ] = plotcnce(log10(setup.Nvec), y);
h{5,1}.Color = [0, 0, 0]; hp{5,1}.Color = [0, 0, 0]; hp{5,2}.Color = [0, 0, 0];
lgd = legend(harr, 'CNCE2', 'CNCE6', 'CNCE20', 'MLE');
setPlotLabels('Sample size $$\log_{10} N$$', '$$\log_{10}$$ sqError', lgd, 'southwest');
ylim(ylimits)
xlim([2, 4.5])
cnceSaveFig(figure_save_folder, savefile_CNCE_consistency)

% NCE consistency
y = log10(cat(3, res.err_nce(:, :, kappa_ind), res.err_mle));
[ h, harr, hp ] = plotcnce(log10(setup.Nvec), y);
h{5,1}.Color = [0, 0, 0]; hp{5,1}.Color = [0, 0, 0]; hp{5,2}.Color = [0, 0, 0];
lgd = legend(harr, 'NCE2', 'NCE6', 'NCE20', 'MLE');
setPlotLabels('Sample size $$\log_{10} N$$', '$$\log_{10}$$ sqError', lgd, 'southwest');
ylim(ylimits)
xlim([2, 4.5])
cnceSaveFig(figure_save_folder, savefile_NCE_consistency)

% CNCE and NCE consistency comparision
y = log10(cat(3, res.err_cnce(:, :, 3, 1), res.err_nce(:, :, 3), res.err_mle));
[ h, harr, hp ] = plotcnce(log10(setup.Nvec), y);
h{4,1}.Color = [0, 0, 0]; hp{4,1}.Color = [0, 0, 0]; hp{4,2}.Color = [0, 0, 0];
lgd = legend(harr, 'CNCE10', 'NCE10','MLE');
setPlotLabels('Sample size $$\log_{10} N$$', '$$\log_{10}$$ sqError', lgd, 'southwest');
ylim(ylimits)
xlim([2, 4.5])
cnceSaveFig(figure_save_folder, savefile_consistency_comparision)


end

function ylimits = getYlim(modelName)

    if strcmp(modelName, 'gauss')
        ylimits = [-3, 1];
    elseif strcmp(modelName, 'ICA')
        ylimits = [-4, 1];
    elseif strcmp(modelName, 'bernoulli')
        ylimits = [-7, -2];
    elseif strcmp(modelName, 'lognormal')
        ylimits = [-5, 0];
    elseif strcmp(modelName, 'ring')
        ylimits = [-4.5, 2];    
    else
        ylimits = [];
    end

end
