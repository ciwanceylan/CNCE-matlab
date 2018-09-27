% --------------------------------------------------
%
% [out1d out2d]=createSmoothLocalGrating(patchsize,freqvalue,orvalue,phasevalue,location)
%
% patchsize : size of the square image patch (scalar)
% freqvalue : frequency of grating : 0 to 0.5 (patchsize/2 cycles per patch)
% orvalue   : orientation of grating where 0 means horizontal "stripes"
% phasevalue: phase of grating where 0 means sinusoidal
% location  : center of the window (1,1) is top left, 
%             (patchsize,patchsize) is bottom right
%
% --------------------------------------------------

function [out1d, out2d] = createSmoothLocalGrating(patchsize, freqvalue, orvalue, phasevalue, location, b)

% Use fixed value for half-response spatial frequency  
% b in (0.4 2.6) for the Monkey
% b = 1.4;

% Use fixed value for aspect ratio gamma
% gamma = 0.6;
% gamma = 1;

sigma = 1/freqvalue *1/pi*sqrt(log(2)/2)*(2^b+1)/(2^b-1);

%create matrices that give the different x and y values
x = 1:patchsize;
y = x;
[xm, ym] = meshgrid(x,y);

%rotate x and y values according to the desired orientation
zm = sin(orvalue).*xm + cos(orvalue).*ym;

%compute sin and cos functions
grating2d=sin(zm*freqvalue*2*pi + phasevalue);

% apply window
r = sqrt((xm - location(1)).^2 + (ym - location(2)).^2);
window = exp(-0.5 * (r/sigma).^2);
winGrating2d = grating2d .* window;

%make average value zero and normalize to unit norm
%av = mean(winGrating2d(:));
%winGrating2d = winGrating2d-av;

%out2d=winGrating2d/(norm(winGrating2d,'fro')+1e-9);

out2d = winGrating2d;
out1d = reshape(out2d, [1, patchsize^2]); %[1 x winSize^2]


return;
