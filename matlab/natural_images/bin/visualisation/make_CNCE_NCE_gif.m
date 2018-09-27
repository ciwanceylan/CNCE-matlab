function make_CNCE_NCE_gif(folderpath)
%% Makes a gif which compares the training results for NCE and CNCE
    % h = figure;
    % axis tight manual
    global CNCE_IMAGE_RESULTS_FOLDER;
    
    filename_CNCE = fullfile(CNCE_IMAGE_RESULTS_FOLDER, 'CNCE_L1.gif');
    filename_NCE = fullfile(CNCE_IMAGE_RESULTS_FOLDER, 'NCE_L1.gif');

    CNCE_files = [folderpath, filesep, 'L1_n100_%d_*.mat'];
    NCE_files = [folderpath, filesep, 'L1_n100_NCE_%d_*.mat'];

    for n = 0:20
        CNCE_file = dir(sprintf(CNCE_files, n));
        load([CNCE_file.folder '/' CNCE_file.name])
        h = plot_rFields_L1(cest, 10, 'low');
        axis tight manual
        title(['CNCE, iteration:' num2str(n)], 'FontSize', 14)
        drawnow

        frame = getframe(h);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);

        if n == 0
            imwrite(imind, cm, filename_CNCE, 'gif', 'Loopcount', inf, 'DelayTime',1.5);
        else
            imwrite(imind, cm, filename_CNCE, 'gif', 'WriteMode', 'append', 'DelayTime',1.5);
        end
    end

    for n = 0:20
        NCE_file = dir(sprintf(NCE_files, n));
        load([NCE_file.folder '/' NCE_file.name])
        h = plot_rFields_L1(ncest, 10, 'low');
        title(['NCE, iteration:' num2str(n)], 'FontSize', 14)
        axis tight manual
        drawnow

        frame = getframe(h);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);

        if n == 0
            imwrite(imind, cm, filename_NCE, 'gif', 'Loopcount', inf, 'DelayTime',1.5);
        else
            imwrite(imind, cm, filename_NCE, 'gif', 'WriteMode', 'append', 'DelayTime',1.5);
        end
    end
end