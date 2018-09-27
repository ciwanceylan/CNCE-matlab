function out = rectifDot(x)
%RECTIFDOT Derivative of linear rectifing function. Either smoothed or not.
% Smooth version: 0.5*tanh(2*x)+0.5;
% Non-smooth version: Heavyside(x)

GLOBAL_SMOOTHRECTIF = true;
if isempty(GLOBAL_SMOOTHRECTIF) || GLOBAL_SMOOTHRECTIF
	out = 0.5*tanh(2*x)+0.5;
else
	out = (x > 0) + 0.01 * (x < 0);
end
end

