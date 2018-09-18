function plot_noise_data_ratio_results(setup, res)
%plot_noise_data_ratio_results Plots the noise-to-data ratio results given 
%the result struct 'res' generated from run_cnce_experiment

global CNCE_RESULTS_FOLDER

% Figure saving setup
figure_save_folder = fullfile(CNCE_RESULTS_FOLDER, 'figures',...
    setup.modelName, 'noise_data_ratio');


ylimits = getYlim(setup.modelName);

for n = 1:length(setup.Nvec)
    err_cnce = sqrt(squeeze(res.err_cnce(:, n, :, 1)));
    err_nce = sqrt(squeeze(res.err_nce(:, n, :)));
    err_mle = sqrt(repmat(res.err_mle(:, n), [1, length(setup.kappaVec)]));
    savefile_consistency_comparision = ['CNCE_NCE_comparision' num2str(setup.D) 'D_' num2str(setup.Nvec(n)) 'N'];

    % CNCE and NCE consistency comparision
    y = log10(cat(3, err_cnce , err_nce, err_mle));
    [ h, harr, hp ] = plotcnce(setup.kappaVec, y);
    h{4,1}.Color = [0, 0, 0]; hp{4,1}.Color = [0, 0, 0]; hp{4,2}.Color = [0, 0, 0];
    lgd = legend(harr, 'CNCE', 'NCE','MLE');
    setPlotLabels('Noise-to-data sample ratio $$(\kappa)$$', '$$\log_{10}$$ sqrError', lgd, 'southwest');
    ylim(ylimits)
    cnceSaveFig(figure_save_folder, savefile_consistency_comparision)


end
end

function ylimits = getYlim(modelName)

    if strcmp(modelName, 'gauss')
        ylimits = [-1.5, 0];
    elseif strcmp(modelName, 'ICA')
        ylimits = [-2, -0.5];
    elseif strcmp(modelName, 'bernoulli')
        ylimits = [-5, 0];
    elseif strcmp(modelName, 'lognormal')
        ylimits = [-5, 0];
    elseif strcmp(modelName, 'ring')
        ylimits = [-5, 2];    
    else
        ylimits = [];
    end

end