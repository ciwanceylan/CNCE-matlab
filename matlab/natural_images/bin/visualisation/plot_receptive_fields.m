function plot_receptive_fields(estimator, Lx)
%PLOT_receptive_fields Visualise the learned recpetive fields for an
%arbitrary layer with default options.
%   Input:
%       estimator: a trained CNCEstimator or NCEstimator
%       Lx: either 'L1', 'L2', 'L3', 'L4', 'all' or a cell with
%       combinations, i.e. {'L1', 'L2'}

if ~isa(Lx, 'cell')
    Lx = cellstr(Lx);
end

if ismember('L1', Lx) || ismember('all', Lx)
    nNeurons = estimator.neuronLayers{1}.layerDimensions(2);
    if nNeurons < 200
        cols = 10;
    else
        cols = ceil(nNeurons/20);
    end
   plot_rFields_L1(estimator, cols, 'low');
end

if ismember('L2', Lx) || ismember('all', Lx)
   plot_rFields_L2(estimator, 20, 'low', 2, 10);
end

if ismember('L3', Lx) || ismember('all', Lx)
    nNeurons = estimator.neuronLayers{2}.layerDimensions(2);
    units = randi(nNeurons, 1, 3); % choose three random units
    plot_rFields_response(estimator, 'L3', length(units), units, 0.1, 'Linear', length(units));
end

if ismember('L4', Lx) || ismember('all', Lx)
    nNeurons = estimator.neuronLayers{2}.layerDimensions(3);
    units = randi(nNeurons, 1, 1); % choose 1 random unit, very slow to plot
    plot_rFields_pooling_L4(estimator, units, 6, 'Linear')
end

end

