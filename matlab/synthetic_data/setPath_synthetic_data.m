function oldPath = setPath_synthetic_data
    %setPath Sets the matlab path for bin
    oldPath = path;

    addiPath = genpath('bin');
    path(addiPath,path);
    

    global CNCE_DATA_FOLDER CNCE_RESULTS_FOLDER
    CNCE_DATA_FOLDER = './data';
    CNCE_RESULTS_FOLDER = './results';

end

