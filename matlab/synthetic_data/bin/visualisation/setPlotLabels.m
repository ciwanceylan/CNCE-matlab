function setPlotLabels( myxlabel, myylabel, lgd, varargin)
%SETPLOTLABELS Help function to set nice looking labels and legend for
%current figure

if (nargin > 3)
	lgdloc = varargin{1};
else
	lgdloc = 'northeast';
end
fig = gcf;
hx = xlabel(myxlabel, 'Interpreter', 'latex');
hy = ylabel(myylabel, 'Interpreter', 'latex');
set(gca,'FontSize',18,'Fontname','Helvetica','Box','off','Tickdir','out','Ticklength',...
	[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',22,'fontname','avantgarde','color',[.3 .3 .3]);
if (~isempty(lgd))
	lgd.FontSize = 14;
	set(lgd, 'Interpreter', 'latex', 'Location', lgdloc );
end
grid on

end

