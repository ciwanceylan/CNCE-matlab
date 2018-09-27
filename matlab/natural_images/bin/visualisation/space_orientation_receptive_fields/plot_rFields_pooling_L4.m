function plot_rFields_pooling_L4(cest, units, maxL3Units, activator, fileName)
	if nargin < 5, fileName = []; end
	if nargin < 4, activator = []; end
    
    percentage = 0.9; % default value 0.9.

	Q = cest.neuronLayers{2}.R.^2;
	nUnits = length(units);
	[val, index] = sort(Q(units,:), 2, 'descend');

	plotStrength = zeros(nUnits, maxL3Units);
	plotIndex = zeros(nUnits, maxL3Units);
	cumVal = cumsum(val, 2);
	cumVal = bsxfun(@rdivide, cumVal, cumVal(:,end));
	
	
	for k = 1:nUnits
		cutIndex = find(cumVal(k,:) > percentage, 1);
		diff = maxL3Units - cutIndex;
		strength = val(k,:) / val(k , 1);
		if diff >= 0
			plotIndex(k, 1:cutIndex) = index(k, 1:cutIndex);
			plotStrength(k, 1:cutIndex) = strength(1:cutIndex);
		else
			plotIndex(k, 1:maxL3Units) = index(k, 1:maxL3Units);
			plotStrength(k, 1:maxL3Units) = strength(1:maxL3Units);
		end
	end
	
	marginX = 0.2;
	marginY = 0.1;
	kthBlue  = [25 , 84 , 166] / 255;
	fileNameEndings = {'Max_rFields', 'Min_rFields', 'Max_icons', 'Min_icons'};
	for k = 1:nUnits
		L3units = plotIndex(k,:);
		L3units(L3units == 0) = [];
		figs = plot_rFields_response(cest, 'L3', maxL3Units, L3units, marginX,...
			activator, [], 'dontshoworsavefigures');
		for ii = 1:length(figs)
			figure(figs{ii});
			ylabel(figs{ii}.Children(end), num2str(units(k)),'Rotation', 0, 'FontSize', 24, ...
				'Fontname', 'Helvetica', 'Color', 0.5*[1 1 1],...
				'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right');
			pos = figs{ii}.Position;
			figWidth = pos(3);
			startX = marginX/(maxL3Units);
			for j = 1:length(L3units)
				barwidth = 1/(maxL3Units) - marginX/(maxL3Units)^2; % Works for some reason....
				%ax = axes('position', [startX, 0, (1 - marginX)/(maxL3Units + 1) , 0.95*marginY], 'visible', 'off', 'box', 'off');
				ax = axes('position', [startX, marginX - marginY, barwidth , 0.95*marginY], 'visible', 'off', 'box', 'off');
				barh(plotStrength(k, j), 'FaceColor', kthBlue)
				xlim([0, 1]);
				ax.Box = 'off';
				ax.XTick = []; ax.XTickLabel = [];
				ax.YTick = []; ax.YTickLabel = [];
				startX = startX + barwidth; 
			end
			if isempty(fileName)
				figs{ii}.Visible = 'on';
			elseif strcmp(fileName, 'dontshoworsavefigures')
				;
			else
				sName = sprintf('%s_%s_u%d', fileName{2}, fileNameEndings{ii}, units(k));
				saveFigure(fileName{1}, sName, 'pdf', figs{ii});
				close(figs{ii});
			end
		end
	end
	
end


