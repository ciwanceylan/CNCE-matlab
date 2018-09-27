function oldPath = setPath_natural_images
    %setPath_natural_images Sets the matlab path for the natural image
    %folder and adds global variables for loading data and saving results
    oldPath = path;

    addiPath = genpath('bin');
    path(addiPath,path);
    
    % Path to folders where data and results are saved
    global CNCE_IMAGE_DATA_FOLDER CNCE_IMAGE_RESULTS_FOLDER
    % NOTE: If you change the data folder make sure to copy the folder
    %   'raw_images' and the subfolder 'tiffImages' with its content to 
    %   the new folder
    CNCE_IMAGE_DATA_FOLDER = './data';
    CNCE_IMAGE_RESULTS_FOLDER = './results';

end

