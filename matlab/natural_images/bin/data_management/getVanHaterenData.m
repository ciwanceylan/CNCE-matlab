function X = getVanHaterenData(samples, winsize)
%getVanHaterenData Gathers image patches from the Van Hateren images.
% The images are downsampled, lightning normalised using the log-transform
%   and finally normalised to unit variance
%   Input:
%       samples            total number of patches to take
%       winsize            patch width in pixels
%   Output:
%       X                  The patches as a [N x winsize^2] matrix
% 

global CNCE_IMAGE_DATA_FOLDER;

CONTROL_PLOT = 0;

%Image database
dirName = fullfile(CNCE_IMAGE_DATA_FOLDER, 'raw_images', 'VanHateren');
dirName = [dirName, filesep];
w = 1536; h = 1024;

%Find images...

d = dir([dirName '*.iml']);
files = {d.name};
imagenum = length(files);
fprintf('Sampling %d images \n', imagenum)

% This will hold the patches
X=zeros(winsize^2,samples);
totalsamples = 0;

% Don't sample too close to the edges
BUFF=4;

% For control plot...
con=ceil(rand*imagenum);

% Determine how many patches to take per image
getsample = max(floor(samples/imagenum),1);
fprintf('Taking %d patches per image \n', getsample)

% Step through the images
for i=1:imagenum

  % Display progress
  fprintf('[%d/%d] ',i,imagenum);
  
  % Load a scene
  f1=fopen([dirName files{i}],'rb','ieee-be');
  I=fread(f1,[w,h],'uint16');
  fclose(f1);
  I = double(I)';
  
  %fprintf('\t Resize, take log and scale image to unit variance \n')
  I = log(imresize(I,0.5,'nearest'));
  I = I/std(I(:));
  
% Control plot
if i==con && CONTROL_PLOT
figure
plot([1 1], [1 size(I,1)])
hold on
plot([size(I,2) size(I,2)], [1 size(I,1)])
plot([1 size(I,2)], [1 1])
plot([1 size(I,2)], [size(I,1) size(I,1)])
title(['Training data control, image [' num2str(con) ']'])
end
  %the last image:the remaining patches...
  if i==imagenum, getsample = samples-totalsamples; end
  
  % Extract patches at random from this image to make data vector X
  for j=1:getsample
    r=BUFF+ceil((size(I,1)-winsize-2*BUFF+1)*rand);
    c=BUFF+ceil((size(I,2)-winsize-2*BUFF+1)*rand);

    if i==con && CONTROL_PLOT 
      plot([c c+winsize], [r r],'--r')
      plot([c c+winsize], [r+winsize r+winsize],'--r')
      plot([c c], [r r+winsize],'--r')
      plot([c+winsize c+winsize], [r r+winsize],'--r')
    end

    totalsamples = totalsamples + 1;
    X(:,totalsamples) = ...
        reshape( I(r:r+winsize-1,c:c+winsize-1),winsize^2,1);
  end


if totalsamples>=samples
  break 
end

end  

fprintf('\n\n');
X = X';
end
