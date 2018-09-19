function setup = config_noise_to_data_ratio(modelName, dim)

isValidModelName(modelName);
[~, dim] = validateDim(modelName, dim);

global CNCE_RESULTS_FOLDER
% Setup
% Reproducability
setup.r = 240;

% Experiment setup
setup.nrDatasets = 100; % No. different parameter sets to use
setup.Nvec = [500]; % Faster than using N=5000, qualitative result the same
% setup.Nvec = [5000]; % Value of N used in the ICML paper
setup.kappaVec = [2, 4, 6, 8, 10, 20, 40, 60]; % Noise-to-data ratio

% Model setup
setup.D = dim;
setup.modelName = modelName;
setup.modelName_NCE = [setup.modelName, '_NCE'];

% Noise setup
setup.mIterMax = 1;
if strcmp(modelName, 'bernoulli')
    setup.noiseFunCNCE = @berNoise;
    setup.noiseFunNCE = @berNCEnoise;
else
    setup.noiseFunCNCE = @gNoise;
    setup.noiseFunNCE = @gNCEnoise;
end

% Optimisation setup
setup.optNCE.alg = 'fminunc';
% setup.optNCE.alg = 'all';  % Slow, but use to avoid issues with NCE
setup.optNCE.maxIter = 6000;
setup.optNCE.verbose = 0;
setup.optCNCE = setup.optNCE;

% setup save folder and file
setup.saveFolder = fullfile(CNCE_RESULTS_FOLDER, setup.modelName, date);
if ~isfolder(setup.saveFolder)
	mkdir(setup.saveFolder)
end
setup.savefile = fullfile(setup.saveFolder, ...
    ['noise_to_data_ratio_' num2str(setup.D) 'D.mat']);

end

function answer = isValidModelName(modelName)

    if (~ismember(modelName, ...
        {'gauss', 'ICA', 'lognormal', 'bernoulli', 'ring'}))
        error(['%s is not implemented. ' ...
            'Please use one of the following: '...
            'gauss, ICA, lognormal, bernoulli, ring'], modelName)
    end
    answer = true;
end

function [answer, dim] = validateDim(modelName, dim)

    if length(dim) > 1 || dim < 1 || ~(floor(dim) == dim)
        error('dim must be a positive integer')
        answer = false;
    end

    if ismember(modelName, {'lognormal', 'bernoulli'}) && dim > 1
        warning('Only dim = 1 is available for lognormal and bernoulli model')
        warning('Setting dim to 1')
        dim = 1;
    end

    answer = true;

end
