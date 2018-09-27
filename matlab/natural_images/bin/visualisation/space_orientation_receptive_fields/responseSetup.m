function [phase, allFreq, nFreqs, b, radius, delta, N, M, allOr] = responseSetup(winSize)
%responseSetup Setup the ranges for the space-orientation receptive field
%parameters
	phase = pi/2;
	allFreq = 0.1:0.05:0.25; % above 0.25, low pass filtering sets in
	nFreqs = length(allFreq);
	b = 1.4;
	[radius, delta, N] = getRadiusAndDelta(allFreq, winSize, b);
	M = 120;       % number of points on the circle, i.e. number of orientations
	allOr = ((0:M) / M) *2*pi; % Orientations in radians, half a turn thanks to symmetry
end