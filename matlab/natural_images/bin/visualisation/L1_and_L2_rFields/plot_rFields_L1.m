function h = plot_rFields_L1(cest, cols, frqType, varargin)
%plot_rFields_L1(cest, cols, frqType, varargin)
% cest        : CNCEstimator/NCEestimator with learned parameters
% cols        : number of colums in the grid 
%frqType      : Either 'low' or 'high'. 'Low' uses the dewhitening matrix
%               for visualisation and emphesises low frequencies. 'High' uses the
%               whitening matrix and emphesises high frquencies
% h           : the figure handle (used in 'make_CNCE_NCE.gif')

dp = cest.dataProcessors{1};
max_gray = 225;

W = cest.neuronLayers{1}.W;

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
%% normalization

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


if nargin > 3
	selected = varargin{1};
	Ivis = Ivis(selected, :);
end
[I, Imaxi, Imini] = make_rFields(Ivis, 1, cols);

% For setting the axis ticks
maxIcons = cols;
nRows = ceil(size(Ivis, 1)/cols);
RI = imref2d(size(I));
RI.XWorldLimits = [1, 2*maxIcons + 1];
RI.YWorldLimits = [1, 2*nRows + 1];

h = figure;
colormap(gray(256));
iptsetpref('ImshowBorder','loose'); 
if exist('parent')
	imshow(I, RI, 'displayrange', [Imini Imaxi], 'parent', parent);
else
  imshow(I, RI, 'displayrange', [Imini Imaxi]);
  truesize;  
end

% Setting the axis ticks
ax = gca;
ax.YTick = 4*[1:nRows/2];
ax.YTickLabel = cellstr(num2str(2*[1:nRows]'))';
ax.XTick = 4*[1:maxIcons/2];
ax.XTickLabel = cellstr(num2str(2*[1:maxIcons]'))';
set(gca,'FontSize',18,'Fontname','Helvetica','Box','off','Tickdir','out','Ticklength',...
	[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
  
end