function download_VanHateren_images()
%DOWNLOAD_VANHATEREN_IMAGES Downloades the VanHatern images used for
%training for the ICML 2018 CNCE paper
% The original images are not available one-by-one, only as a zip file,
% which can be found here: http://pirsquared.org/research/vhatdb/full/ 
% This function currently downloads 11 random images from the 99 available
% one-by-one.

global CNCE_IMAGE_DATA_FOLDER;

fprintf('Starting download of Van Hateren images...\n')
save_folder = fullfile(CNCE_IMAGE_DATA_FOLDER, 'raw_images', 'VanHateren');
if ~isfolder(save_folder)
    mkdir(save_folder);
end

url_base = 'http://pirsquared.org/research/vhatdb/';

% These files cannot be downloaded individually
van_hateren_files_imcl = {'imk00017.iml',...
                         'imk00034.iml'....
                         'imk00041.iml',...
                         'imk00192.iml',...
                         'imk00195.iml',...
                         'imk00344.iml',...
                         'imk00389.iml',...
                         'imk00753.iml',...
                         'imk00825.iml',...
                         'imk03658.iml',...
                         'imk03691.iml'};

nFiles = length(van_hateren_files_imcl);

random_numbers = randperm(90) + 9;
random_numbers = random_numbers(1:11);

van_hateren_files = {nFiles};
for i = 1:nFiles
    van_hateren_files{i} = ['imk000' num2str(random_numbers(i)) '.iml'];
end


for i = 1:nFiles
    url = [url_base, van_hateren_files{i}];
    filename = fullfile(save_folder, van_hateren_files{i});
    outfilename = websave(filename,url);
    fprintf('%s saved to disk\n', outfilename)
end
fprintf('done!\n');


end

