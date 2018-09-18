function [ thetaStar, varargout ] = ica_fast_mle(X, thetaIni)
%ICA_FAST Fast MLE for the ICA model
%   See eq. (21) in "Fast and Robust Fixed-Point Algorithms for 
%   Independent Component Analysis", 
%   A. Hyvearinen (1999) for gardient expression 

persistent matversion

nargoutchk(1, 3);
if (isempty(matversion))
	matversion = version('-date');
	matversion = str2num(matversion(end-4:end));
end

if (matversion > 2015)
	% 2016 option
	options = optimoptions('fminunc','Display', 'off', 'Algorithm','trust-region',...
		'SpecifyObjectiveGradient',true);
else 
	% 2015b options
	options = optimoptions('fminunc', 'Display', 'off', 'Algorithm', 'trust-region',...
		'GradObj', 'on');
end

[N, D] = size(X);
M = length(thetaIni);
nrSources = M/D;
maxIter = 6000;

[thetaStarFmin, fvalFmin, exitflag, output] = fminunc(@nestedMinFun,thetaIni,options);
objfun = @(theta) nestedMinFun(theta');
[thetaStarMini, fvalMini, i] = minimize(thetaIni', objfun, maxIter);

if (fvalMini < fvalFmin)
	thetaStar = thetaStarMini';
	if (nargout > 1)
		% return the value of the objective function at thetaStar
		varargout{1} = fvalMini(end);
	end
	if (nargout == 3)
		aux.optFun = 'minimize';
		aux.fval = fvalMini;
		aux.i = i;
		varargout{2} = aux;
	end
else
	thetaStar = thetaStarFmin;
	if (nargout > 1)
		% return the value of the objective function at thetaStar
		varargout{1} = fvalFmin;
	end
	if (nargout == 3)
		aux.optFun = 'fminunc';
		aux.output = output;
		aux.exitflag = exitflag;
		varargout{2} = aux;
	end
end

function [cost, gradient] = nestedMinFun(theta)

    B = reshape(theta, nrSources, D); % B is "long"
    Shat = X * B'; %[N x nrSources]
	% J = data term + partition function
    J = mean(-1 * sqrt(2) * sum(abs(Shat), 2)) - (nrSources/2)*log(2) + log(abs(det(B')));
    %J = -1 * sqrt(2) * sum(sum(abs(Shat), 2))  + nrSources*log(abs(det(B')));
    cost = -1 * J;
	
	g = -1*sqrt(2)*sign(Shat); % [N x nrSources]
    grad = (eye(D) + g'*Shat/N) * B; % natural gradient
    gradient = -reshape(grad, 1, nrSources * D );
end

end

