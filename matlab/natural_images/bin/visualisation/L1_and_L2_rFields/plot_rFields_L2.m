function varargout = plot_rFields_L2(cest, nUnits, frqType, cols, maxL1Units)
%plot_rFields_L2 Plots the 1st layer receptive fields according to the 2nd
%layer learned pooling.
%
% cest        : CNCEstimator with parameters
% nUnits      : Number of cells to analyze (random selection,
%               if nUnits == 'all', we show all units,
%               if nUnits = [nb1 nb2 ...], we show the units nb1, nb2, ...
% frqType     : Either 'low' or 'high'. 'Low' uses the dewhitening matrix
%               for visualisation and emphesises low frequencies. 'High' uses the
%               whitening matrix and emphesises high frquencies
% cols        : number of columns
% maxL1Units  : max number of L1 cells in each column

percentage = 0.9; % (default: 0.9) Display as many units as to account for this fraction of the norm of the W2 vector
threshold = 0; % (default: 0) Remove units accounting for less than this fraction

% Extract the required matricies
dp = cest.dataProcessors{1};
max_gray = 225;

W = cest.neuronLayers{1}.W;
Q = cest.neuronLayers{1}.R.^2;

dim = size(W, 2);
myNorm = sqrt(sum(W.^2, 2));
Wnorm = bsxfun(@times, W, 1./myNorm);
if strcmp(frqType, 'high')
    Wh = dp.getWh();
    Ir = Wnorm * Wh';
elseif strcmp(frqType, 'low')
    deWh = dp.getdeWh();
    Ir = Wnorm * deWh;
else
    error('frqType should be high or low')
end

% Normalization

% remove dc (actually not needed since dc is by construction zero)
av = mean(Ir,2);
max(abs(av));
Ir = bsxfun(@plus,Ir,-av);

% switch sign so that max value is positive
% ok since models are symmetric in x

maxi = max(abs(Ir), [], 2);
maxi2 = max(Ir, [], 2);
mysign = maxi==maxi2; 
mysign = 2*(mysign-0.5);

Ir = bsxfun(@times,Ir,mysign);

% set scale
mini = min(Ir, [], 2);
Ir2 = bsxfun(@plus,Ir,-mini);
Ivis = bsxfun(@times,Ir2,max_gray./(maxi-mini));

% Different plotting cases
if isa(nUnits,'char') && strcmp(nUnits,'all')
  nUnits = size(Q, 1);
  selectedUnits = 1:nUnits;
elseif length(nUnits) > 1
  selectedUnits = nUnits;
  nUnits = length(nUnits);
else
  tmp = randperm(size(Q, 1));
  selectedUnits = tmp(1:nUnits);
%   fprintf(['Selected units: \n']);
%   display(selectedUnits)
end
% L2

% sort the weights
[val, index] = sort(Q(selectedUnits,:), 2, 'descend');

cumVal = cumsum(val, 2);
cumVal = bsxfun(@rdivide, cumVal, cumVal(:,end));

% remove those units which have a maximal Q-value less than treshold
remIndex = find(max(val, [], 2) < threshold);
val(remIndex,:) = [];
cumVal(remIndex,:) = [];
index(remIndex,:) = [];
selectedUnits(remIndex) = [];
nUnits = size(index,1);

% plot the units given the pooling information
plotIt(Ivis, nUnits, maxL1Units, threshold, percentage, cumVal, val, index, cols);
set(gcf, 'visible', 'on')

if nargout ==1
    varargout{1} = selectedUnits;
end

% fprintf('Showed units: \n')
% display(selectedUnits)

end

function plotIt(feature, nUnits, maxL1Units, threshold, percentage, cumVal, val, index, cols)
%plotIt Plots the features together with the pooling strengths.
  
	% add a blank-feature
	featurePlot = [feature; 255*ones(1, size(feature,2))];
	dIndex = size(featurePlot, 1);
	plotIndex = zeros(nUnits, maxL1Units+1);
	plotStrength = zeros(nUnits, maxL1Units+1);

	kk=1;
	for k=1:nUnits
	  
	  if val(k, 1) >= threshold
		  
		  % find p-percentage
		  [cutIndex, value]=min(find(cumVal(k,:) > percentage));
		  diff = maxL1Units - cutIndex;
		  
		  % normalize the strength
		  strength = val(k,:) / val(k,1); 
		  if diff >= 0
			  plotIndex(kk, 1:maxL1Units) = [index(k, 1:cutIndex), dIndex * ones(1, diff)];
			  plotStrength(kk, 1:maxL1Units) = [strength(1:cutIndex), zeros(1, diff)];
		  else
			  plotIndex(kk, 1:maxL1Units) = index(k, 1:maxL1Units);
			  plotStrength(kk, 1:maxL1Units) = strength(1:maxL1Units);
		  end
		  
		  plotIndex(kk, end) = dIndex;
		  plotStrength(kk, end) = 0;
		  kk = kk + 1;
	  end
	  
	end

	plotIndex = reshape(plotIndex',[],1);
	plotStrength = reshape(plotStrength',[],1);
	[I, Imaxi, Imini] = make_rFields(featurePlot(plotIndex, :), 1, cols*(maxL1Units+1), plotStrength);
	
	% For setting the axis ticks
	maxIcons = cols*(maxL1Units+1);
	nRows = ceil(nUnits/cols);
	RI = imref2d(size(I));
	RI.XWorldLimits = [1, 2*maxIcons + 1];
	RI.YWorldLimits = [1, 2*nRows + 1];

	figure('visible', 'off')
	colormap(gray(256));
	iptsetpref('ImshowBorder','loose'); 
	if exist('parent')
		imshow(I, RI,'displayrange', [Imini Imaxi], 'parent', parent);
    else
	  imshow(I, RI, 'displayrange', [Imini Imaxi]);
	  truesize;  
	end
	
	% Setting the axis ticks
	ax = gca;
	ax.YTick = 4*[1:nRows/2];
	ax.YTickLabel = cellstr(num2str(2*(1:nRows)'))';
	ax.XTick = 4*[1:maxIcons/2];
	ax.XTickLabel = cellstr(num2str(2*(1:maxIcons)'))';
	set(gca,'FontSize',14,'Fontname','Helvetica','Box','off','Tickdir','out','Ticklength',...
		[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
end