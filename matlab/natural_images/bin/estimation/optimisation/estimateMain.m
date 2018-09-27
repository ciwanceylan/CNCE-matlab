function [ thetaStar, varargout ] = estimateMain( objFun, thetaInit, opt)
%ESTIMATEMAIN Wraper for the optimisation functions.
%   Input:
%       objFun: Function handle for function to optimise
%       thetaInit: initialisation for theta
%       opt: optimisation options

nargoutchk(1, 2);
persistent matversion

% Because different matlab versions have been use we need to check if we are using 2015 or later.
if (isempty(matversion))
	matversion = version('-date');
	matversion = str2num(matversion(end-4:end));
end

% if (isempty(opt))
	% %opt.objFun = @(thetap)lossFun(theModel, thetap, U, logP, kappa);
	% opt.alg = 'minimize';
	% opt.maxIter = 400;
	% opt.verbose = 0;
% end
if (opt.verbose)
	fprintf('Starting optimisation... \n')
end
if (strcmp(opt.alg, 'fminunc') || strcmp(opt.alg, 'all'))
	if (opt.verbose)
	dispOpt = 'Iter';
	else
	dispOpt = 'off';
	end
	if (matversion > 2015)
	% 2016 option
	options = optimoptions('fminunc','Display', dispOpt, 'Algorithm','quasi-newton', ...
		'SpecifyObjectiveGradient', true, 'MaxIterations', opt.maxIter);
	else 
	% 2015b options
	options = optimoptions('fminunc', 'Display', dispOpt, 'Algorithm', 'quasi-newton', ...
		'GradObj', 'on', 'MaxIter', opt.maxIter);
	end
	
	[thetaStar_fmin, fval_fmin] = fminunc(objFun, thetaInit, options);
end
if (strcmp(opt.alg, 'minimize') || strcmp(opt.alg, 'all'))
	
	% objFunT = @(thetap)objFun(thetap');
	[thetaStar_mini, fval_mini, i] = minimize(thetaInit, objFun, opt.maxIter, opt.verbose);
end

if ( strcmp(opt.alg, 'fminunc'))
	thetaStar = thetaStar_fmin;
	aux.loss = fval_fmin;
	aux.alg = 'fminunc';
elseif (strcmp(opt.alg, 'minimize'))
	thetaStar = thetaStar_mini;
	aux.loss = fval_mini;
	aux.alg = 'minimize';
	aux.nIter 	= i;

elseif (strcmp(opt.alg, 'all'))
	if (fval_mini < fval_fmin)
		thetaStar = thetaStar_mini;
		aux.loss = fval_mini;
		aux.alg = 'minimize';
		aux.nIter 	= i;
	else
		thetaStar = thetaStar_fmin;
		aux.loss = fval_fmin;
		aux.alg = 'fminunc';
	end
else
	error('Unknown opt algorithm.')
end

if (nargout > 1)
	% return the value of the objective function at thetaStar
	varargout{1} = aux;
end