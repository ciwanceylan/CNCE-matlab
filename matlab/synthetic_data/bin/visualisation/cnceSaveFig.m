function cnceSaveFig( foldername, filename, varargin )
%CNCESAVEFIG Wraper for matlabs print function used to save figures

if (nargin > 2)
	fig = figure(varargin{1});
else
	fig = gcf;
end

set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize', [pos(3), pos(4)]);
if ~isfolder(foldername)
	mkdir(foldername)
end
savefile = fullfile(foldername, filename);
print(fig, savefile, '-dpdf', '-r0')
print(fig, savefile, '-dpng', '-r0')

end

