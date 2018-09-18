function prmtrs = generate_parameters(modelName, nDataSets, dim)
%GENERATE_PARAMETERS Generates ground truth parameters 
%   Ground truth parameters are generated and saved.
%   Generation procedure depends on model.

global CNCE_DATA_FOLDER
filename = fullfile(CNCE_DATA_FOLDER, 'gt_parameters',...
    [modelName '_parameters']);
[filepath,~,~] = fileparts(filename);
if ~isfolder(filepath)
    mkdir(filepath)
end

fprintf('Generating %s parameters...', modelName)
if strcmp(modelName, 'gauss')
    filename = [filename, '_', num2str(dim) 'D'];
    prmtrs.theta_gt = cell(nDataSets, 1);
    prmtrs.theta_init = cell(nDataSets, 1);
    prmtrs.c_gt = cell(nDataSets, 1);
    prmtrs.c_init = cell(nDataSets, 1);
    prmtrs.E = cell(nDataSets, 1);
    prmtrs.D_flat = cell(nDataSets, 1);
    for k=1:nDataSets
        % Ground truth, ensure positive definite matrix
        M = randn(dim, dim);  % a random matrix
        % Turn random matrix into a orthonormal eigenvectros
        prmtrs.E{k} = (M * M')^(-0.5) * M;     % A^(-0.5) = U D^(-0.5) U' if A = UDU'
        % generate radnom eigenvalues for the covariance matrix
        tmp = 0.1 + 0.9 * rand(dim, 1);
        prmtrs.D_flat{k} = sort(tmp, 1, 'descend');
        % Calculate the precision matrix as the invers of the covariance matrix
        Lambda_gt = prmtrs.E{k} * diag(prmtrs.D_flat{k}.^(-1)) * prmtrs.E{k}';
        prmtrs.theta_gt{k} = LambdatoTheta(Lambda_gt);
        prmtrs.c_gt{k} = -dim / 2 * log(2*pi) - 0.5 * sum(log(abs(prmtrs.D_flat{k})));
        

        % Initialisation
        U = randn(dim, dim);
        U = (U * U')^(-0.5) * U;
        tmp = 0.1 + 0.9 * rand(dim,1);
        tmp = sort(tmp, 1, 'descend');
        Lambda_init = U * diag(tmp.^(-1)) * U';
        prmtrs.theta_init{k} = LambdatoTheta(Lambda_init);
        prmtrs.c_init{k} = randn;
    end
    
elseif strcmp(modelName, 'ICA')
    filename = [filename, '_', num2str(dim) 'D'];
    prmtrs.theta_gt = cell(nDataSets, 1);
    prmtrs.theta_init = cell(nDataSets, 1);
    prmtrs.c_gt = cell(nDataSets, 1);
    prmtrs.c_init = cell(nDataSets, 1);
    prmtrs.A = cell(nDataSets, 1);
    nSources = dim;
    maxCondi = 10;
    for k=1:nDataSets
        % Ground truth
        % random mixing matrix with sufficient condition number
        condA = 1e6;
        while (condA > maxCondi)
          prmtrs.A{k} = randn(dim, dim);
          condA = cond(prmtrs.A{k});
        end
        Btrue = pinv(prmtrs.A{k});  % if a is [D x Ms] the B is [Ms x D ]
        prmtrs.theta_gt{k} =  reshape(Btrue, [1, nSources * dim]);  % [nrSources x D] -> [1 x nSources * D]
        prmtrs.c_gt{k} = log(abs(det(Btrue))) - 0.5 * log(2) * dim;    % waring, not need to change for nrSources != D

        % Initialisation
        Binit = 0.1 * randn(dim * dim, 1);
        prmtrs.theta_init{k} = Binit';
        prmtrs.c_init{k} = randn;
    end

elseif strcmp(modelName, 'lognormal')
    
    % Ground truth
    tmp = 0.1 + 0.9 * rand(nDataSets, 1); 
    prmtrs.theta_gt = [1./tmp, zeros(nDataSets, 1)];
    sigma = 1./sqrt(prmtrs.theta_gt(:,1));
	prmtrs.c_gt = -log(sigma * sqrt(2*pi));
    
    % Initialisation
    tmp = 0.1 + 0.9 * rand(nDataSets, 1); 
    prmtrs.theta_init = [1./tmp, randn(nDataSets, 1)];
    prmtrs.c_init = zeros(nDataSets, 1);

elseif strcmp(modelName, 'bernoulli')

    theta = rand(nDataSets, 1);
    limit = 0.01;
    I = find(abs(theta - 0.5) > (0.5 - limit));

    while (sum(I) > 0)
        theta(I) = rand(length(I), 1);
        I = find(abs(theta - 0.5) > (0.5 - limit));
    end
    prmtrs.theta_gt = [theta, 1 - theta];
    prmtrs.theta_init = 5 * rand(nDataSets, 2) + 0.1;
    
    prmtrs.c_gt = zeros(nDataSets, 1);
    prmtrs.c_init = zeros(nDataSets, 1);
    
elseif strcmp(modelName, 'ring')
    filename = [filename, '_', num2str(dim) 'D'];
    mu = 5 * rand(nDataSets, 1) + 5;
    sig = 1.2 * rand(nDataSets, 1) + 0.3;
    prec = 1 ./ sig.^2;
    prmtrs.c_gt = -0.5 * log(2*pi) - log(sig);
    prmtrs.theta_gt = [mu, prec];
 
    mu_ini = 2 * rand(nDataSets, 1) + 6;
    sig_ini = 1.2 * rand(nDataSets, 1) + 0.3;
    prmtrs.c_init = randn(nDataSets, 1);
    prmtrs.theta_init = [mu_ini, 1 ./ sig_ini.^2];
    
else
	error('Parameters generation does not exists for desired model')
end

save(filename,'prmtrs');
fprintf('done!\n')
end

