function X = getTiffPatches(samples, winsize) 
%getTiffPatches Gathers image patches from the Tiff images.
% No pre-processing of the images
%   Input:
%       samples            total number of patches to take
%       winsize            patch width in pixels
%   Output:
%       X                  The patches as a [N x winsize^2] matrix
% 
    
global CNCE_IMAGE_DATA_FOLDER

% Path to the data 
loadDir = fullfile(CNCE_IMAGE_DATA_FOLDER, 'raw_images', 'tiffImages');
loadDir = [loadDir, filesep];
d=dir([loadDir '*.tiff']);
files={d.name};
imagenum=length(files);
fprintf(['Extracting patches from ' num2str(imagenum) ' images \n'])
getsample = ceil(samples/imagenum);

% This will hold the patches
X=zeros(winsize^2,samples);
totalsamples = 0;

% Don't sample too close to the edges
BUFF=4;

% Sampling

for k=1:imagenum

    fileName = [loadDir files{k}];
    I = imread(fileName); 

    %the last image:the remaining patches
    if k==imagenum
        getsample = samples-totalsamples; 
    end

    fprintf(['Extract ' num2str(getsample) ' patches from image ' num2str(k) '\n'])

    % Extract patches at random from this image to make data vector X
    for j=1:getsample
        r=BUFF+ceil((size(I,1)-winsize-2*BUFF+1)*rand);
        c=BUFF+ceil((size(I,2)-winsize-2*BUFF+1)*rand);

        totalsamples = totalsamples + 1;
        X(:,totalsamples) = ...
            reshape( I(r:r+winsize-1,c:c+winsize-1),winsize^2,1);
    end

    if totalsamples>=samples
        break 
    end      
end  

fprintf('\n');
X = X';
end  