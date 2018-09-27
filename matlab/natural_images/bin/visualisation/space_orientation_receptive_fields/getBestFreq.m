function [maxValue, bestFreqIndex] = getBestFreq(resp)
%getBestFreq Get the Gabor filter frequency for each unit which produces the
%maximal response .
%
% resp:             cell array of length nFreqs. Each entry is a 
%                   4-dimensional array (x,y,orientation,unit)
%                   containing each units response to a Gabor filter
%                   centered at (x,y) and with given orientation
% maxValue:         maximal response for each unit
% bestFreqIndex:    the frequency index giving the maximal response	for
%                   each unit
	nVectors = size(resp{1});
	if length(nVectors) < 4
		nVectors = 1;
	else
		nVectors = nVectors(end);
	end
	nFreqs = length(resp);

    maxi_freq = zeros(nFreqs, nVectors );
    % iterate over all frequencies
    for k = 1:nFreqs
        tmp = resp{k};
        % Get the maximal reponse for each unit given the frequency
        for kk = 1:nVectors
            tmp2 = squeeze(tmp(:, :, :, kk)); % x, y, orientation, unit
            maxi_freq(k, kk) = max(tmp2(:));
            clear tmp2;
        end
        clear tmp;
    end
    [maxValue, bestFreqIndex] = max(maxi_freq, [], 1);
end