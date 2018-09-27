function figs = plot_rFields_response(cest, Lx, cols, units, margin, activator,	kFigMax, fileName)
	%plot_rFields_response Visualises the response to Gabor stimulation of the 'Lx' neuron layer
    % using both space-orientation receptive fields and data patches
    % producing a large response
    %
	%	cest - CNCE or NCEstimator
	%	Lx - either 'L2', 'L3' or 'L4'
	% 	cols - the number of columns in the mosaic visualisation
	%	units - vector specifying which units in the layer to visualise
	%	margin - margin between tiles in the mosaic. Values 0.05, 0.1 and 0.2 works well
	%	activator - name of activator to use for visualisation
	%	kFigMax - maximum number of tiles in a single figure. Default is length(units)
	%	fileName - If specified the visualisations will be saved under this file name

	[layer, dimInx] = setLayer(Lx);
	if nargin < 4 || isempty(units)
		nUnits = cest.neuronLayers{layer}.layerDimensions(dimInx);
		units = 1:nUnits;
	else
		nUnits = length(units);
	end
	if nargin < 5 || isempty(margin)
		margin = 0;
	end
	if nargin < 6, activator = []; end
	if nargin < 7 || isempty(kFigMax), kFigMax = nUnits; end % max units per figure
	if nargin < 8, fileName = []; end

    % Both the maximal and minimal response can be plotted
	fileNameEndings = {'Max', 'Min'};
	
	% Get response for patches
	dataset = 'Tiff';
	X = getRawData(dataset, cest.dataOpt.winsize, 10000, 42, false);
	[Zmax, Zmin] = response(X, cest, Lx, 'relative_to_mean', 0, activator);
	[~, Imaxs] = sort(Zmax, 1, 'descend');
	[~, Imins] = sort(Zmin, 1, 'descend');
	
	% Screen info for fig size
	screenInfo = get(0, 'Screensize');
	screenHight = screenInfo(end);
	winSize = cest.dataOpt.winsize;
	subfigSize  = 150;
	rows = ceil(kFigMax/cols);
	marginPx 	= margin * subfigSize;
	figHight 	= subfigSize * rows;
	figWidth 	= subfigSize * cols;
	marginX 	= marginPx / figWidth;
	marginY 	= marginPx / figHight;
	figHight 	= (1 + marginY) * figHight;
	figWidth 	= (1 + marginX) * figWidth;

    % Extract the response from the learned network
	[respMax, respMin] = spaceOrRF(cest, 0, Lx, activator);
    
    % Setup the variables needed to make pretty plot
    [~, ~, ~, ~, radius, delta, ~, ~, allOr] = responseSetup(winSize);
    % Unless the minimum response is all zeros, we plot it 
	plotMin = 1 *(sum(respMin{1}(:)) + sum(respMin{end}(:)) ~= 0); % Unless no response (all zero)
    % Store figures in cells
	figs = cell(1+plotMin, 1);
	figsIcons = cell(1+plotMin, 1);
    
    % Iterate over maximal and minimal response plot 
	for ii = 0:plotMin
		if ii == 0, resp = respMax; else, resp = respMin; end
        
        % Get the frequency producing the largest response for each unit
		[~, bestFreqIndex] = getBestFreq(resp);

		newFigs = true;
		nk = 0;
        % Iterate over all the units and plot each of them on the grid
		for k = 1:nUnits
            if newFigs
				[figs{ii+1}, figsIcons{ii + 1}] = openNewFigs(ii, figWidth, figHight, screenHight);
				newFigs = false;
            end
            % Space-orientation receptive field plotting
            % select the rField figure
			figure(figs{ii+1}); 
			% pick best frequency
			tmp = resp{bestFreqIndex(k)}; % the result for the best frequency
			myResp = tmp(:,:,:,units(k)); % for unit k
			clear tmp
			myRadius = radius(bestFreqIndex(k)); % corresponding radius
			myDelta = delta(bestFreqIndex(k)); % corresponding delta
			fprintf('%d ', units(k))
            
            % Setup the ploting location on the grid
			row = floor(nk/cols);
			col = nk - row * cols;
			scaleX = (1-marginX)/cols; scaleY = (1-marginY)/rows;
			startX = (1-marginX) * col/cols + marginX; startY = 1 - scaleY - (1-marginY) * row/rows; 
			ax = axes('position', [startX, startY, scaleX, scaleY ], 'Box', 'on' );
			ax.XTick = []; ax.XTickLabel = [];
			ax.YTick = []; ax.YTickLabel = [];
			ax.LineWidth = 2;
			if ii == 1, ax.XColor = 'r'; ax.YColor = 'r'; end
            
            % Plot the response for each location and orientation
			plotOrientationMap_grid(allOr, myResp, myRadius, myDelta, winSize, startX, startY, scaleX, scaleY);
			
            % Patches with large response plotting
			figure(figsIcons{ii+1}); % Select the icons (patches) figure
			ax = axes('position', [startX, startY, 0.95*scaleX, 0.95*scaleY ]);
            if ii == 0
                plotIcons_grid(X(Imaxs(1:25, units(k)), :), 5, ax, 'k');
            else
                plotIcons_grid(X(Imins(1:25, units(k)), :), 5, ax, 'r');
            end
            
            % Use nk to determine if new figure is needed (to avoid too
            % many RFs in a single figure
			nk = nk + 1;
			if nk == kFigMax || k == nUnits
				if isempty(fileName) 
					figs{ii+1}.Visible = 'on';
					figsIcons{ii+1}.Visible = 'on';
				elseif strcmp(fileName, 'dontshoworsavefigures')
					if nargout < 1
						close(figs{ii+1});
						close(figsIcons{ii+1});
					end
				else
					sName = sprintf('%s_%s_u%d_%d', fileName{2}, fileNameEndings{ii+1},...
						units(k-nk+1), units(k));
					pause(1) %This is for some reason necessary to avoid a visual bug when saving...
					saveFigure(fileName{1}, [sName, '_rFields'], 'pdf', figs{ii+1});
					pause(1) % Same here........
					saveFigure(fileName{1}, [sName, '_icons'], 'pdf', figsIcons{ii+1});
					if nargout < 1
						close(figs{ii+1});
						close(figsIcons{ii+1});
					end
				end
				newFigs = true;
				nk = 0;
			end
		end
		fprintf('\n')
	end
	if nargout > 0
		figs = cat(1, figs, figsIcons);
	end
end

function [layer, dimInx] = setLayer(Lx)
%setLayer Parse the Lx input, setting other options accordningly
	if strcmp(Lx, 'L1')
		error('Not all functions currently implemented for L1')
	end
	% NeuronLayer
	if strcmp(Lx, 'L1') || strcmp(Lx, 'L2')
		layer = 1;
	elseif strcmp(Lx, 'L3') || strcmp(Lx, 'L4')
		layer = 2;
	else
		error('Unrecognized neuron layer')
	end

	% Layer dimmension index
	if strcmp(Lx, 'L1') || strcmp(Lx, 'L3')
		dimInx = 2;
	elseif strcmp(Lx, 'L2') || strcmp(Lx, 'L4')
		dimInx = 3;
	else
		error('Unrecognized neuron layer')
	end
	
end

function plotOrientationMap_grid(or, respIn, radius, delta, winSize, startX, startY, scaleX, scaleY)
%plotOrientationMap_grid Plots the orientation response for each position
%on a square grid using mmpolar

	color = [0.9 0.9 0.9];
	%axis image
	N = size(respIn,1);  % number of positions along x or y-axis where
					   % local gratings are applied
	maxResp = max(abs(respIn(:)));
	fprintf(['Maximal value: ' num2str(maxResp) '\n'])

	Rmax = 1;
% 	handler = zeros(N,N);
	for x = 0:1:N-1
		for y=0:1:N-1
            % Get the response and normalise
			resp = squeeze(respIn(x+1,y+1,:))/maxResp;

            % Setup for mmpolar
			location = [radius + x*delta, radius + y*delta];
			left = scaleX * (location(1)-radius)/winSize + startX;
			bottom = startY + scaleY *(winSize-location(2)-radius)/winSize;
			width = scaleX * 2*radius/winSize;
			height = scaleY * 2*radius/winSize;
			axes('Position',[left, bottom, width, height],'visible','off', 'Box', 'off' );
			% mmpolar is very slow....
			mmpolar(or,resp,'b-',...
                   'RLimit',[0 Rmax],...
                   'grid','off','TTickLabelVisible','off', 'RTickLabelVisible','off',...
                   'backgroundcolor','none',...
                   'bordercolor',color, 'linewidth', 1);
			hold on
		end
	end
end

function [fig, figsIcon] = openNewFigs(ii, figWidth, figHight, screenHight)
	fig = figure('position', [ii*figWidth, screenHight - figHight - 80, figWidth, figHight],...
		'color', 'w', 'visible', 'on');
	figsIcon = figure('position', [ii*figWidth, screenHight - figHight - 80, figWidth, figHight],...
		'color', 'w', 'visible', 'on');
end

function plotIcons_grid(Ir, cols, parent, borderColor)
%plotIcons_grid Plots the patches (icons) giving a large response on a grid 
	max_gray = 225;

	% set scale
	maxi = max(abs(Ir), [], 2);
	mini = min(Ir, [], 2);
	Ir2 = bsxfun(@plus,Ir,-mini);
	Ivis = bsxfun(@times,Ir2,max_gray./(maxi-mini));
	[I, Imaxi, Imini] = make_rFields(Ivis, 1, cols);
	
	% For setting the axis ticks
	maxIcons = cols;
	nRows = ceil(size(Ivis, 1)/cols);
	RI = imref2d(size(I));
	RI.XWorldLimits = [1, 2*maxIcons + 1];
	RI.YWorldLimits = [1, 2*nRows + 1];
	
	colormap(gray(256));
	iptsetpref('ImshowBorder','loose'); 
	imshow(I, RI, 'displayrange', [Imini Imaxi], 'parent', parent);
	set(gca, 'XTick', [], 'YTick', [], 'XTickLabel', [], 'YTickLabel', [], 'Box', 'on',...
					'LineWidth', 2, 'XColor', borderColor, 'YColor', borderColor )

end