function out = rectif(x)
%RECTIF Linear rectifing function. Either smoothed or not.
% Smooth version: log(cosh(2*x)) + 0.5*x + 0.17;
% Non-smooth version: max(x, 0)

GLOBAL_SMOOTHRECTIF = true;
if isempty(GLOBAL_SMOOTHRECTIF) || GLOBAL_SMOOTHRECTIF
	% Note that 0.5*1./alpha*log(cosh(alpha*x))+0.5*x = -0.1733 for x->-inf and
	% alpha = 2 (which was chosen below); 
	% Since log(cosh(1000)) = log(cosh(-1000)) = Inf, we need to manually set
	% the values for large absolute inputs
	tmp = x>100;  
	out = (0.5*1/2*log(cosh(2*x))+0.5*x+0.17);
	out(tmp) = x(tmp);
	tmp = x<-100;
	out(tmp) = -0.0033;
else
	out = max(x, 0.01*x);
end
end

