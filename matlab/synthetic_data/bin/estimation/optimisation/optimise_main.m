function [ thetaStar, varargout ]  = optimise_main( objFun, thetaInit, opt)
%optimise_main Wraper for optimisation.
%   Either Matlabs optimisation code (based on Quasi-Newton) or 
%   minimize.m can be used.
%   Input:
%       objFun - function handle to objective function to be optimised
%       thetaInit - the initial value for theta
%       opt - optimisation options
%   Output:
%       thetaStar - final value for theta
%       vargout{1}:
%           aux.loss - final value of loss

nargoutchk(1, 2);
persistent matversion

% Because different matlab versions have been use we need to check if we are using 2015 or 2016.
if (isempty(matversion))
	matversion = version('-date');
	matversion = str2num(matversion(end-4:end));
end

if (isempty(opt))
    % Default options
	opt.alg = 'fminunc';
	opt.maxIter = 400;
	opt.verbose = 0;
end
if (opt.verbose)
	fprintf('Starting optimisation... \n')
end
if (strcmp(opt.alg, 'fminunc') || strcmp(opt.alg, 'all'))
    % Use matlabs optimisation
	if (opt.verbose)
	dispOpt = 'Iter';
	else
	dispOpt = 'off';
	end
	if (matversion > 2015)
	% 2016 option
	options = optimoptions('fminunc','Display', dispOpt,...
        'Algorithm','quasi-newton', ...
		'SpecifyObjectiveGradient', true, 'MaxIterations', opt.maxIter);
	else 
	% 2015b options
	options = optimoptions('fminunc', 'Display', dispOpt,...
        'Algorithm', 'quasi-newton', ...
		'GradObj', 'on', 'MaxIter', opt.maxIter);
	end
	
	[thetaStar_fmin, fval_fmin] = fminunc(objFun, thetaInit, options);
end

if (strcmp(opt.alg, 'minimize') || strcmp(opt.alg, 'all'))
    % Use minimize.m
	objfunT = @(thetap)objFun(thetap');
	[thetaStar_mini, fval_mini, i] =...
        minimize(thetaInit', objfunT, opt.maxIter, opt.verbose);
	fval_mini = fval_mini(end);
end

% Optional outputs
if ( strcmp(opt.alg, 'fminunc'))
    % matlabs optimisation
	thetaStar = thetaStar_fmin;
	aux.loss = fval_fmin;
	aux.alg = 'fminunc';
elseif (strcmp(opt.alg, 'minimize'))
    % minimize.m
	thetaStar = thetaStar_mini';
	aux.loss = fval_mini;
	aux.alg = 'minimize';
	aux.nIter 	= i;

elseif (strcmp(opt.alg, 'all'))
    % Use both minimize and matlabs optimisation
	if (fval_mini < fval_fmin)
		thetaStar = thetaStar_mini';
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