function [ X, theta_gt, theta_init, varargout] = ...
    getData( N, dim, modelName, nDataSets, varargin )
%GETDATA Help function to get data using pre-generated parameters.

if nargin > 4
    dataSetNr = varargin{1};
else
    dataSetNr = randi(nDataSets, 1);
end

if nargin > 5
	seed = varargin{2};
    rng(seed);
end


global CNCE_DATA_FOLDER
modelNameParts = strsplit(modelName, '_');
if strcmp(modelNameParts{1}, 'bernoulli') ||...
        strcmp(modelNameParts{1}, 'lognormal')
        filename = fullfile(CNCE_DATA_FOLDER, 'gt_parameters',...
        [modelNameParts{1} '_parameters.mat']);
else
    filename = fullfile(CNCE_DATA_FOLDER, 'gt_parameters',...
        [modelNameParts{1} '_parameters_' num2str(dim) 'D.mat']);

end

prmtrs = getParameters(filename, nDataSets, modelNameParts{1}, dim);

if strcmp(modelName, 'gauss') 
	theta_gt = prmtrs.theta_gt{dataSetNr};
	theta_init = prmtrs.theta_init{dataSetNr};
    
	data = randn(dim, N);
	data = prmtrs.E{dataSetNr} * diag(prmtrs.D_flat{dataSetNr}.^0.5) * data;
	X = data';

elseif (strcmp(modelName, 'gauss_NCE'))
	theta_gt = [prmtrs.theta_gt{dataSetNr}, prmtrs.c_gt{dataSetNr}];
	theta_init = [prmtrs.theta_init{dataSetNr}, prmtrs.c_init{dataSetNr}];
    
	data = randn(dim, N);
	data = prmtrs.E{dataSetNr} * diag(prmtrs.D_flat{dataSetNr}.^0.5) * data;
	X = data';

elseif strcmp(modelName, 'ICA')
	theta_gt = prmtrs.theta_gt{dataSetNr};
	theta_init = prmtrs.theta_init{dataSetNr};
    
	b = 1/sqrt(2);
	U = rand(N, dim) - 0.5;
	S = b * sign(U) .* log(1-2*abs(U));
	aux.Smean = mean(S);
	aux.Scov = cov(S);
	X = S * prmtrs.A{dataSetNr}';

elseif strcmp(modelName, 'ICA_NCE')
	theta_gt = [prmtrs.theta_gt{dataSetNr}, prmtrs.c_gt{dataSetNr}];
	theta_init = [prmtrs.theta_init{dataSetNr}, prmtrs.c_init{dataSetNr}];
    
	b = 1/sqrt(2);
	U = rand(N, dim) - 0.5;
	S = b * sign(U) .* log(1-2*abs(U));
	aux.Smean = mean(S);
	aux.Scov = cov(S);
	X = S * prmtrs.A{dataSetNr}';

elseif strcmp(modelName, 'lognormal')
	theta_gt = prmtrs.theta_gt(dataSetNr, :);
	theta_init = prmtrs.theta_init(dataSetNr, :);

	pd = makedist('Lognormal','mu', 0,'sigma', 1 / sqrt(theta_gt(1)));
	X = random(pd, N, 1);
    
elseif strcmp(modelName, 'lognormal_NCE')
	theta_gt = [prmtrs.theta_gt(dataSetNr, :), prmtrs.c_gt(dataSetNr)];
	theta_init = [prmtrs.theta_init(dataSetNr, :), prmtrs.c_init(dataSetNr)];

	pd = makedist('Lognormal','mu', 0,'sigma', 1 / sqrt(theta_gt(1)));
	X = random(pd, N, 1);

elseif strcmp(modelName, 'bernoulli')
    theta_gt = prmtrs.theta_gt(dataSetNr, :);
	theta_init = prmtrs.theta_init(dataSetNr, :);
    
	X = rand(N, 1) < theta_gt(2);

elseif (strcmp(modelName, 'bernoulli_NCE'))
    theta_gt = [prmtrs.theta_gt(dataSetNr, :), prmtrs.c_gt(dataSetNr)];
	theta_init = [prmtrs.theta_init(dataSetNr, :), prmtrs.c_init(dataSetNr)];
    
	X = rand(N, 1) < theta_gt(2);

elseif (strcmp(modelName, 'ring'))
    theta_gt = prmtrs.theta_gt(dataSetNr, :);
	theta_init = prmtrs.theta_init(dataSetNr, :);
    theta_init(1) = theta_gt(1);  % Set the mean to the ground truth
    
    X = genRingData(N, dim, theta_gt(1), 1./sqrt(theta_gt(2)), 0.1);
  
elseif (strcmp(modelName, 'ring_NCE'))
    theta_gt = [prmtrs.theta_gt(dataSetNr, :), prmtrs.c_gt(dataSetNr)];
	theta_init = [prmtrs.theta_init(dataSetNr, :), prmtrs.c_init(dataSetNr)];
    theta_init(1) = theta_gt(1);  % Set the mean to the ground truth
    
    X = genRingData(N, dim, theta_gt(1), 1./sqrt(theta_gt(2)), 0.1);

else
	error('Could not find data of desired type and dimension')
end

if (nargout > 3)
	aux.dataSetNr = dataSetNr;
	varargout{1} = aux;
end


end

function prmtrs = getParameters(filename, nDataSets, modelName, dim)
% Helper function which generates a new parameter set if one doesn't 
%   already exists
    
    if ~isfile(filename)
        prmtrs = generate_parameters(modelName, nDataSets, dim);
    else
        load(filename, 'prmtrs');
        if size(prmtrs.theta_gt, 1) < nDataSets
            prmtrs = generate_parameters(modelName, nDataSets, dim);
        end
    end
end
