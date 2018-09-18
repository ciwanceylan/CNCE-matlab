function [ thetaStar, history, varargout ] = ...
    optimisation_whistory( U, thetaInit, theModel, logPyx_ratio)
%optimisation_whistory Wraper for optimisation which also outputs intermediate iterations.
%   Either Matlabs optimisation code (based on Quasi-Newton) or 
%   minimize.m can be used.
%   Input:
%       U - [X Y1 Y2 ... Ykappa] [N, D, kappa + 1] matrix containing data and noise
%		thetaInit - initial parameter values.
%       theModel - a function handle to the model. The model should take
%       	return outputs and take inputs on the form presented in the Model template file.     
%		logPyx_ratio - [N x kappa] the log ratio of the conditional noise pdf.
%   Output:
%       thetaStar - final value for theta
%       history - cell contraining the values of theta for every iteration
%           and the corresponding loss
%       vargout{1}:
%           aux.loss - final value of loss
%		aux - struct containing information of the optimoption result

nargoutchk(1, 3);
persistent matversion

% Because different matlab versions have been use we need to check if we are using 2015 or 2016.
if (isempty(matversion))
	matversion = version('-date');
	matversion = str2num(matversion(end-4:end));
end

% run non-linear optimoption

if (matversion > 2015)
	% 2016 option
	options = optimoptions('fminunc','Display', 'off',...
        'Algorithm','quasi-newton','SpecifyObjectiveGradient',true, ...
        'OutputFcn', @myoutput);
else
	% 2015b options
	options = optimoptions('fminunc', 'Display', 'off',...
        'Algorithm', 'quasi-newton', 'GradObj', 'on', ...
        'OutputFcn', @myoutput);
end


history = cell(2, 1);
objfun = @(theta)cnce_loss(theModel, theta, U, logPyx_ratio);

function stop = myoutput(theta, optimvalues, state)
	stop = false;
	if isequal(state, 'iter')
		history{1} = [history{1}; theta];
		history{2} = [history{2}; optimvalues.fval];
	end
end

[thetaStar,fval,exitflag,output] = fminunc(objfun, thetaInit, options);

if (nargout > 2)
	% return the value of the objective function at thetaStar
	varargout{1} = fval;
end
if (nargout == 4)
	aux.output = output;
	aux.exitflag = exitflag;
	varargout{2} = aux;
end
end