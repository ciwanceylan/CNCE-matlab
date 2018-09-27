% --------------------------------------------------
%  
% varargout = spaceOrRF_L4(cest, intervention, PLOT, Lx, selected)
%  [respL3, bestFreqIndex, maxValuePerUnit, respL3v, respL2, respL2u]
% cest: a CNCEstimator object containing the estimation
% intervention: 0 (for nothing), 'noInhibition' to 
%               remove negative 3rd layer weights
% PLOT     : 1 for plotting figures, 0 else
% varargout: Several computed quantities (see code)
% 
% --------------------------------------------------


function [respMax, respMin, totMeanMax, totMeanMin] = spaceOrRF(cest, intervention, Lx, activator)
	if nargin < 4, activator = []; end
	winSize = cest.dataOpt.winsize;

	[phase, allFreq, nFreqs, b, radius, delta, N, M, allOr] = responseSetup(winSize); % defines phase, allFreq, nFreqs, b, radius, delta, N, M, allOr
	
	if strcmp(Lx, 'L2')
		nUnits = cest.neuronLayers{1}.layerDimensions(3);
	elseif strcmp(Lx, 'L3')
		nUnits = cest.neuronLayers{2}.layerDimensions(2);
	elseif strcmp(Lx, 'L4')
		nUnits = cest.neuronLayers{2}.layerDimensions(3);
	else
		error('Unrecognized Lx %s', Lx)
	end
	
	respMax = cell(1, nFreqs);
	respMin = cell(1, nFreqs);
	
	totResponseMax = 0;
	totResponseMin = 0;
	for kFreq = 1:length(allFreq)
		freq = allFreq(kFreq);
		myN = N(kFreq);
		myDelta = delta(kFreq);
		myRadius = radius(kFreq);

		% model layer outputs
		respMax_tmp = -ones(myN, myN, M + 1, nUnits);% location, location, orientation, units
		% model outputs, inhibition
		respMin_tmp = -ones(myN, myN, M + 1, nUnits);% location, location, orientation, units
		
		for x = 0:1:myN-1
			for y = 0:1:myN-1
				% handle all units for all orientations at location (x,y) at the
				% same time
				location = [myRadius + x*myDelta, myRadius + y*myDelta]; 
				
				tmpI = zeros(M, winSize^2); % Stim image
				for k = 0:M
					or = allOr(k+1); % orientation in radians
					[grating, ~] = createSmoothLocalGrating(winSize, freq, or, phase, location, b);      
					% normalize grating to be between +/- 100
					[maxi, index] = max(abs(grating)); % Get maximal absolute value
					grating = sign(grating(index)) * grating; % make largest absolute value positive
					mini = min(grating); % Get smallest value 
					grating = (grating - mini) / (maxi - mini); % Normalize to between 0 and 1
					grating = grating - 0.5;
					grating = grating * 200; % between -100 and 100
					tmpI(k+1, :) = grating;
					
					% figure(100)
					% sIm = imresize(reshape(grating, 32, 32), 2);
					% imagesc(sIm); colormap gray; colorbar; truesize;
					% headline = sprintf('%.3f, %d, %d, %.3f', freq, x, y, or);
					% title(headline)
					% pause(0.01)
				end
				[Ymax, Ymin] = response(tmpI, cest, Lx, 'relative_to_mean', intervention, activator);
				fprintf('.')
				
				respMax_tmp(x+1,y+1,:,:) = Ymax;
				respMin_tmp(x+1,y+1,:,:) = Ymin;
				
				clear Ymax Ymin;
			end
		end
		respMax{kFreq} 	= respMax_tmp;
		respMin{kFreq} 	= respMin_tmp;
		totResponseMax = totResponseMax + sum(respMax_tmp(:));
		totResponseMin = totResponseMin + sum(respMin_tmp(:));
		fprintf('\n')
	end
	% Calculate and remove mean stimulation
	totMeanMax = totResponseMax / sum(N.^2 .* (M +1) * nUnits);
	totMeanMin = totResponseMin / sum(N.^2 .* (M +1) * nUnits);
	
	% for kFreq = 1:length(allFreq)
	%	respMax{kFreq} = respMax{kFreq}(:, :, :, units) - totMeanMax;
	%	respMin{kFreq} = respMin{kFreq}(:, :, :, units) - totMeanMin;
	% end

end 