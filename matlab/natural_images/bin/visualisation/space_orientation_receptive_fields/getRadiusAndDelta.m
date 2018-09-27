function [radius, delta, N] = getRadiusAndDelta(allFreq, winSize, b)
%getRadiusAndDelta(allFreq,winSize)
% Maps spatial frequency to radius and spacing on the 
% grid where test local gratings are placed.
    
    % radius of "circle" on the grid, equal to the std of the Gaussian
    % window of the Gabor stimulus
    radius = 1./allFreq*1/pi*sqrt(log(2)/2)*(2^b+1)/(2^b-1); 
    delta = zeros(1, length(allFreq));
                                                             
    N = max(2,ceil(winSize./(2*radius)));
    for k=1:length(allFreq)
        tmp = linspace(radius(k), winSize-radius(k), N(k));
        delta(k)=tmp(2)-tmp(1);
    end
end