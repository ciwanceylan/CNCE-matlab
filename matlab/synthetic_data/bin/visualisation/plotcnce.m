function [ h, harr, hp ] = plotcnce( X, Y, varargin )
%PLOTSQERR Summary of this function goes here
%   Detailed explanation goes here

narginchk(2, 4);
if (nargin > 2)
	fig = varargin{1};
else
	fig = figure;
end


[d1, d2, d3] = size(Y);
if (d3 > 5 && d1 > 1)
	error('Too many lines in same plot!')
end
figure(fig);
hold on
h = cell(d3, 2);
hp = cell(d3, 2);

markers = {'-o', '-s', '-d', '-*', '-v'};
markersDoted = {'--o', '--s', '--d', '--*',  '--v'};
markersMedian = {'--o', '--d', '--s', '--^'};
markersMean = {'-o', '-d', '-s', '-^'};
addColors
colors = {c.kthBlue, c.kthRed, c.niceOrange, c.kthGreen, c.kthLightBlue, c.kthYellow, [0,0,0]};
colors2 = {c.kthBlue, c.kthBlue, c.kthRed, c.kthRed, c.kthBlue, c.kthBlue, c.kthRed, c.kthRed };



for k = 1:d3
    if (d1 == 1)
		% The loss fucntion
		if (size(X, 3) > 1)
			h{k} = plot(X(1,:,k), Y(1,:,k), 'LineWidth', 1.3);
		else
			h{k} = plot(X, Y(1,:,k), 'LineWidth', 1.3);
		end
		if (k < 7)
			h{k}.Color = colors2{k};
			if (mod(k,2) == 0)
				h{k}.LineStyle = '-';
				h{k}.Marker = markersMean{k/2}(end);
			else
				h{k}.LineStyle = '--';
				h{k}.Marker = markersMedian{(k+1)/2}(end);
			end
		end
	else
		aVal = 0.8;
		col = colors{k};
		mark = markers{k};
		markDot = markersDoted{k};
		if k == d3, col = colors{end};end
		if k == d3, mark = markers{end};end
		if k == d3, markDot = markersDoted{end};end
		Ys = sort(Y(:,:,k), 1);
		Ymed = median(Ys, 1);
		Ymean = mean(Ys, 1);
        Ystd = std(Ys, 1);
		q01 = floor(0.1*d1);
		q09 = ceil(0.9*d1);
		Y01 = Ys(q01,:);
		Y09 = Ys(q09, :);
		if (d3 < 3)
			e01 = (Ymed - Y01)';
			e09 = (Y09 - Ymed)';
			if (sum(e01.^2 + e09.^2) == 0)
			h{k, 1} = plot(X, Ymed, markersMean{k});
			else
			[h{k, 1}, hp{k, 1}] = boundedline(X, Ymed, [e01, e09], markersMedian{k}, 'alpha');
			h{k, 2} = plot(X, Ymean,  markersMean{k}, 'LineWidth', 1.3 );
			hp{k, 1}.FaceColor = col;
			h{k, 2}.Color = col;
			end
			h{k, 1}.LineWidth = 1.3;
			h{k, 1}.Color = col;
		else
			
			hp{k, 1} = plot(X, Y01, markDot, 'Color', [col, aVal], ...
				'MarkerFaceColor', col, 'LineWidth', 0.5, 'MarkerSize', 4);
			hp{k, 2} = plot(X, Y09, markDot, 'Color', [col, aVal],...
				'MarkerFaceColor', col, 'LineWidth', 0.5, 'MarkerSize', 4);
			h{k, 1} = plot(X, Ymed, mark, 'Color', col, ...
				'MarkerFaceColor', col, 'LineWidth', 1.6);
%             h{k, 1} = plot(X, Ymean, mark, 'Color', col, ...
% 				'MarkerFaceColor', col, 'LineWidth', 1.6);
        end
    end
end


harr = zeros(2*d3, 1);
for i = 1 : d3*2
	if (~isempty(h{i}))
		harr(i) = h{i};
	end
end

harr = harr(harr ~= 0);
% hx = xlabel(myxlabel);
% hy = ylabel(myylabel);
% set(gca,'FontSize',18,'Fontname','Helvetica','Box','off','Tickdir','out','Ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
% set([hx; hy],'fontsize',22,'fontname','avantgarde','color',[.3 .3 .3]);
% grid on;
end