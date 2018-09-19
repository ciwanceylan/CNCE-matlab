function res = run_cnce_experiment(setup) 
%run_cnce_experiment Runs the experiment detailed by the struct setup.
%   The setup struct can be obtained by running one of the experiment
%   config scripts

[setup.theModelCNCE, setup.Mcnce] = getModel(setup.modelName, setup.D);
[setup.theModelNCE, setup.Mnce] = getModel(setup.modelName_NCE, setup.D);

KN = length(setup.Nvec);
Kkappa = length(setup.kappaVec);

% Setting seeds for reproducability of every iteration of the
% outer loops
rng(setup.r);
setup.noiseSeed = randi(1000);
setup.dataSeed = randi(1000);
setup.dataSeeds = setup.dataSeed + (1:setup.nrDatasets);
setup.noiseSeeds = setup.noiseSeed + (1:setup.nrDatasets);


% data storage
res.err_mle 	= zeros(setup.nrDatasets, KN, 1, 1);
res.time_mle 	= zeros(setup.nrDatasets, KN, 1, 1);
res.err_cnce    = zeros(setup.nrDatasets, KN, Kkappa, setup.mIterMax);
res.epsilon_cnce = zeros(setup.nrDatasets, KN, Kkappa, setup.mIterMax);
res.loss_cnce 	= zeros(setup.nrDatasets, KN, Kkappa, setup.mIterMax);
res.time_cnce 	= zeros(setup.nrDatasets, KN, Kkappa, setup.mIterMax);
res.mIter_cnce 	= zeros(setup.nrDatasets, KN, Kkappa, 1);
res.err_nce 	= zeros(setup.nrDatasets, KN, Kkappa, 1);
res.loss_nce 	= zeros(setup.nrDatasets, KN, Kkappa, 1);
res.time_nce 	= zeros(setup.nrDatasets, KN, Kkappa, 1);

% Iterate over different parameter sets
for dsnr = 1:setup.nrDatasets
	
	fprintf('Starting dataset %d \n', dsnr);
	% pre-generate the noise
	noiseBase = gNoiseBase(max(setup.Nvec), setup.D, max(setup.kappaVec),...
        setup.noiseSeeds(dsnr));
    % Iterate over dataset size
	for kn = 1:KN
		fprintf(['Starting dataSize ' num2str(setup.Nvec(kn))]);
		N = setup.Nvec(kn);
		[ X, thetaTrueNCE, thetaInitNCE] =...
            getData(N, setup.D, setup.modelName_NCE, setup.nrDatasets,...
            dsnr, setup.dataSeeds(dsnr));
		thetaTrueCNCE = thetaTrueNCE(1:end-1);
		thetaInitCNCE = thetaInitNCE(1:end-1);
		
		%------------------------------------o---------------------------------------
		% Calcluate MLE
		tic
        thetaMLE = getMLE(X, thetaInitCNCE, setup.modelName);
        res.time_mle(dsnr, kn, 1, 1) = toc;
        res.err_mle(dsnr, kn, 1, 1) = calculateError(thetaTrueCNCE,...
            thetaMLE, setup.modelName);

		%------------------------------------o---------------------------------------
		% create a base for epsilon
		epsilonBase = gEpsilonBaseFun( X );
		
		for kkap = 1:Kkappa
			kappa = setup.kappaVec(kkap);
			fprintf('.');
            % CNCE
			[theta, aux] = cnce(X, setup.theModelCNCE, thetaInitCNCE, ...
                epsilonBase, noiseBase(1:N,:,1:kappa), setup.noiseFunCNCE, ...
                setup.mIterMax - 1, setup.optCNCE);
			res.time_cnce		(dsnr, kn, kkap, :) = aux.time;
			res.err_cnce		(dsnr, kn, kkap, :) = ...
				calculateError( thetaTrueCNCE, theta, setup.modelName);
			res.epsilon_cnce	(dsnr, kn, kkap, :) = aux.epsilon;
			res.loss_cnce		(dsnr, kn, kkap, :) = aux.loss;
			res.mIter_cnce	(dsnr, kn, kkap, 1) = aux.mIter;
			
            % NCE
			[theta, aux] = nce(X, setup.theModelNCE, thetaInitNCE, ...
                setup.noiseFunNCE, kappa, setup.optNCE);
			res.time_nce	(dsnr, kn, kkap, 1) = aux.time;
			res.err_nce	(dsnr, kn, kkap, 1) = ...
				calculateError(thetaTrueNCE(1:end-1), theta(1:end -1),...
                setup.modelName); %CNCE datatype but remove c from thetaNCE
			res.loss_nce(dsnr, kn, kkap, 1) = aux.loss;
		end
		fprintf('\n')
	end
	fprintf('\n')
	fprintf('Saving...')
	save(setup.savefile, 'setup', 'res')
	fprintf('Done!\n\n')
end

end


